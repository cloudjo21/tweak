from abc import abstractmethod
from copy import deepcopy
from typing import Dict

# from tweak.dataset.multitask_dataset import MultitaskDatasetBuilder
from tweak.metrics.task2metric import MetricBuilder
from tweak.preprocess.converters import ConverterFactory
from tweak.task.task_set import (
    TaskSet,
    TaskType,
)


class ResourceBuilder:

    def __init__(self, task_set: TaskSet):
        self.task_set = deepcopy(task_set)
        # self.mt_dataset_builder = MultitaskDatasetBuilder(self.task_set)
        # self.converter_set = self._create_converters()
        # mt_dataset = self.mt_dataset_builder.mt_dataset
        # self.mt_dataset_builder(mt_dataset, self.converter_set)

    def _create_converters(self):
        mt_dataset = self.mt_dataset_builder.mt_dataset
        converter_set = dict()
        for task_name, dataset in mt_dataset.items():
            # self.phase_datasets[task_name] = dict()
            task = dataset.task
            converter = ConverterFactory.create(task)
            converter.labelize(dataset)
            converter_set[task_name] = converter

        return converter_set

    @property
    def train_dataset(self):
        return self.mt_dataset_builder.get_train_dataset()

    @property
    def validation_dataset(self):
        return self.mt_dataset_builder.get_validation_dataset()

    @property
    def test_dataset(self):
        return self.mt_dataset_builder.get_test_dataset()

    @property
    def model_types(self):
        model_types = dict()
        for task_name, task_type in zip(self.task_set.names, self.task_set.types):
            model_types[task_name] = TaskType.task_type2model_type(task_type)
        return model_types

    @property
    @abstractmethod
    def model_configs(self):
        pass

    @property
    def all_label_list(self):
        label_lists = dict()
        for task_name in self.converter_set:
            label_lists[task_name] = self.converter_set[task_name].label_set()
        return label_lists

    @property
    def compute_metrics(self):
        # TODO composite metric for a specific multi-task
        # e.g., token_classification + sequence_classification
        # TODO generalize the class design of label_set for each task type
        compute_metrics_dict = MetricBuilder.create(
            taskset=self.task_set,
            label_list_dict=self.all_label_list
        )

        return compute_metrics_dict