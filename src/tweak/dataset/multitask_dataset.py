from copy import deepcopy
from datasets import load_dataset, load_from_disk, set_caching_enabled
from torch.utils.data.dataset import Dataset
from typing import Dict

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig

from tweak.dataset.resource_builder import ResourceBuilder
from tweak.metrics.task2metric import MetricBuilder
from tweak.preprocess.converters import ConverterFactory, Converter
from tweak.task.task_set import (
    Task,
    TaskSet,
    TaskType,
    TaskDefaultArrowDatasetColumns
)

set_caching_enabled(False)

class TaskSpecificDataset:
    def __init__(self, task: Task):
        self.task = task
#        self.dataset = load_dataset(
#            # path=task.dataset_name,
#            # data_dir=task.dataset_path,
#            path=task.dataset_name,
#            data_dir=task.dataset_path,
#            download_mode="force_redownload",
#            # TODO support options download_mode="reuse_cache_if_exists" to Task
#        )
        self.dataset = load_from_disk(task.dataset_path)
        self.dataset.cleanup_cache_files()

    @property
    def task_name(self):
        return self.task.task_name

    def items(self):
        return self.dataset.items()


class MultitaskDataset:

    def __init__(self, task_set: TaskSet):
        self.datasets = dict()
        for task in task_set.tasks:
            self.datasets[task.task_name] = TaskSpecificDataset(task)
        self.batch_size = task_set.training_args.per_device_train_batch_size

    def __getitem__(self, task_name) -> TaskSpecificDataset:
        return self.datasets[task_name]

    def items(self):
        return self.datasets.items()


class MultitaskDatasetBuilder:
    """
    provide the data sets for each training phase(train/validation/test)
    """

    def __init__(self, task_set: TaskSet):
        # load dataset for task_set
        self.mt_dataset = MultitaskDataset(task_set)
        self.phase_datasets = None
    
    def __call__(
        self,
        mt_dataset: MultitaskDataset,
        converter_set: Dict[str, Converter]
    ):
        # datasets for each phase from themselves
        self.phase_datasets = dict()
        for task_name, dataset in mt_dataset.items():
            self.phase_datasets[task_name] = dict()
            converter = converter_set[task_name]
            model_columns = TaskDefaultArrowDatasetColumns.get(
                dataset.task
            )

            # phase: train, validation, test
            for phase, phase_dataset in dataset.items():
                self.phase_datasets[task_name][phase] = phase_dataset.map(
                    lambda examples: converter.convert(examples),
                    batched=True,
                    batch_size=mt_dataset.batch_size,
                    load_from_cache_file=False
                )

                # set columns to dataset
                self.phase_datasets[task_name][phase].set_format(
                    type="torch",
                    columns=model_columns
                )

    def get_train_dataset(self):
        train_dataset = {
            task_name: dataset["train"] for task_name, dataset
            in self.phase_datasets.items()
        }
        return train_dataset

    def get_validation_dataset(self):
        validation_dataset = {
            task_name: dataset["validation"] for task_name, dataset
            in self.phase_datasets.items()
        }
        return validation_dataset

    def get_test_dataset(self):
        test_dataset = {
            task_name: dataset["test"] for task_name, dataset
            in self.phase_datasets.items()
        }
        return test_dataset


class MultitaskResourceBuilder(ResourceBuilder):

    def __init__(self, task_set: TaskSet):
        super().__init__(task_set)
        self.mt_dataset_builder = MultitaskDatasetBuilder(self.task_set)
        self.converter_set = self._create_converters()
        mt_dataset = self.mt_dataset_builder.mt_dataset
        self.mt_dataset_builder(mt_dataset, self.converter_set)

        # self.task_set = deepcopy(task_set)
        # self.mt_dataset_builder = MultitaskDatasetBuilder(self.task_set)
        # self.converter_set = self._create_converters()
        # mt_dataset = self.mt_dataset_builder.mt_dataset
        # self.mt_dataset_builder(mt_dataset, self.converter_set)

    # def _create_converters(self):
    #     mt_dataset = self.mt_dataset_builder.mt_dataset
    #     converter_set = dict()
    #     for task_name, dataset in mt_dataset.items():
    #         # self.phase_datasets[task_name] = dict()
    #         task = dataset.task
    #         converter = ConverterFactory.create(task)
    #         converter.labelize(dataset)
    #         converter_set[task_name] = converter

    #     return converter_set

    # @property
    # def train_dataset(self):
    #     return self.mt_dataset_builder.get_train_dataset()

    # @property
    # def validation_dataset(self):
    #     return self.mt_dataset_builder.get_validation_dataset()

    # @property
    # def test_dataset(self):
    #     return self.mt_dataset_builder.get_test_dataset()

    @property
    def model_types(self):
        model_types = dict()
        for task_name, task_type in zip(self.task_set.names, self.task_set.types):
            model_types[task_name] = TaskType.task_type2model_type(task_type)
        return model_types

    @property
    def model_configs(self):
        model_configs = dict()
        for task in iter(self.task_set):
            model_configs[task.task_name] = self._get_model_config(task, self.converter_set)
        return model_configs

    def _get_model_config(self, task: Task, converter_set: Dict[str, Converter]) -> PretrainedConfig:

        # in the case of downstream task
        model_config = AutoConfig.from_pretrained(
            task.pretrained_model_name,
            finetuning_task=task.task_name,
            num_labels=len(converter_set[task.task_name].label2id),
            problem_type=task.problem_type
        )

        return model_config

    # @property
    # def all_label_list(self):
    #     label_lists = dict()
    #     for task_name in self.converter_set:
    #         label_lists[task_name] = self.converter_set[task_name].label_set()
    #     return label_lists

    # @property
    # def compute_metrics(self):
    #     # TODO composite metric for a specific multi-task
    #     # e.g., token_classification + sequence_classification
    #     # TODO generalize the class design of label_set for each task type
    #     compute_metrics_dict = MetricBuilder.create(
    #         taskset=self.task_set,
    #         label_list_dict=self.all_label_list
    #     )

    #     return compute_metrics_dict
