from itertools import chain
from typing import Optional
from torch.utils.data import Dataset

from tunip.constants import OUTSIDE
from tweak.task.task_set import Task, TaskType


class LabelBuilder:
    def __init__(self, label_column_name: str):
        self.label_column_name = label_column_name


class TokenClassificationLabelBuilder(LabelBuilder):
    def __init__(self, label_column_name: str, source_bio_from_dataset: bool):
        super(TokenClassificationLabelBuilder, self).__init__(
            label_column_name
        )

        self.from_bio = source_bio_from_dataset

    def __call__(self, dataset: Dataset):
        all_label_list = list()
        for phase, phase_dataset in dataset.items():
            label_list = phase_dataset[self.label_column_name]
            # TODO option to exclude label_list for test dataset
            if self.from_bio is False:
                label_list = [[f'B-{label}', f'I-{label}'] for labels in iter(label_list) for label in labels]
            all_label_list.extend(label_list)
        all_label_list.append([OUTSIDE])

        unique_labels = set()
        for label in all_label_list:
            unique_labels = unique_labels | set(label)
        unique_label_list = list(unique_labels)
        unique_label_list.sort()

        label2id = {label: i for i, label in enumerate(unique_label_list)}

        return unique_label_list, label2id


class SequenceClassificationLabelBuilder(LabelBuilder):
    def __init__(self, label_column_name: str, problem_type: Optional[str]):
        super(SequenceClassificationLabelBuilder, self).__init__(label_column_name)
        self.problem_type = problem_type

    def __call__(self, dataset):
        all_label_list = list()
        for phase, phase_dataset in dataset.items():
            label_list = phase_dataset[self.label_column_name]
            if self.problem_type == 'multi_label_classification':
                labels_flatten = []
                for i, ll in enumerate(label_list):
                    labels_flatten.extend(ll)
                    if i % 100 == 0:
                        labels_flatten = list(set(labels_flatten))
                all_label_list.extend(list(set(labels_flatten)))

            else:
                # for 'single_label_classification'
                # TODO exclude label_list for test dataset
                all_label_list.extend(label_list)

        unique_labels = set(all_label_list)
        unique_label_list = list(unique_labels)
        unique_label_list.sort()

        label2id = {label: i for i, label in enumerate(unique_label_list)}

        return unique_label_list, label2id


class LabelBuilderFactory:
    @classmethod
    def create(cls, task: Task):
        label_column_name = task.label_column_name
        task_type = task.task_type
        if task_type == TaskType.TOKEN_CLASSIFICATION:
            return TokenClassificationLabelBuilder(
                label_column_name, task.source_bio_from_dataset
            )
        elif task_type == TaskType.SEQUENCE_CLASSIFICATION:
            return SequenceClassificationLabelBuilder(label_column_name, task.problem_type)
        else:
            raise ValueError(f"TaskType:{task_type} is not supported!")
