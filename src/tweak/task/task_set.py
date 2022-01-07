import transformers

from dataclasses import dataclass
from enum import Enum
from itertools import filterfalse
from typing import List, Optional


class TaskType(Enum):
    TOKEN_CLASSIFICATION = 0
    SEQUENCE_CLASSIFICATION = 1
    QUESTION_ANSWERING = 2

    def describe(self):
        return self.name, self.value

    @classmethod
    def task_type2model_type(cls, task_type):
        return cls.model_name2model_type(task_type.describe()[0])

    @classmethod
    def model_name2model_type(cls, model_name):
        model_name = model_name.replace("_", "").upper()
        if "TOKENCLASSIFICATION" in model_name:
            return transformers.AutoModelForTokenClassification
        elif "SEQUENCECLASSIFICATION" in model_name:
            return transformers.AutoModelForSequenceClassification
        elif "QUESTIONANSWERING" in model_name:
            return transformers.AutoModelForQuestionAnswering
        else:
            raise NotSupportedTaskException()


class NotSupportedTaskException(Exception):
    pass


class InputColumnType(Enum):
    TEXT = 0
    TOKENS = 1

    def describe(self):
        return self.name, self.value


class TaskDefaultArrowDatasetColumns:
    @classmethod
    def get(cls, task_type: TaskType):
        if task_type == TaskType.TOKEN_CLASSIFICATION:
            return ["input_ids", "attention_mask", "labels"]

        elif task_type == TaskType.SEQUENCE_CLASSIFICATION:
            return ["input_ids", "attention_mask", "labels"]

        elif task_type == TaskType.QUESTION_ANSWERING:
            return ["input_ids", "attention_mask", "start_positions", "end_positions"]
        raise NotSupportedTaskException


@dataclass
class InputColumn:
    type_: InputColumnType
    name: str


@dataclass
class Task:
    task_name: str
    task_type: TaskType
    dataset_name: str
    dataset_path: str
    pretrained_model_name: str
    input_columns: List[InputColumn]
    label_column_name: str = None
    label_names: list = None
    max_length: int = 32
    source_bio_from_dataset: bool = False

    def get_input_column_by(self, column_type: InputColumnType, column_name: Optional[str] = None) -> Optional[InputColumn]:
        target_input_columns = list(filter(lambda c: c.type_ == column_type, self.input_columns))
        if column_name:
            input_column: Optional[InputColumn] = list(filter(lambda c: c.name == column_name, target_input_columns))[0]
        else:
            input_column: Optional[InputColumn] = target_input_columns[0]
        return input_column
    
    def has_input_column_type(self, column_type: InputColumnType) -> bool:
        input_columns: Optional[list] = list(filter(lambda c: c.type_ == column_type, self.input_columns)) or None
        return input_columns is not None


class TaskSet:
    def __init__(self, tasks: List[Task], service_repo_dir, user_name, domain_name, snapshot_dt, training_args=None):
        self.tasks = tasks
        self.service_repo_dir = service_repo_dir
        self.user_name = user_name
        self.domain_name = domain_name
        self.snapshot_dt = snapshot_dt
        self.training_args = training_args

    def __iter__(self):
        return iter(self.tasks)

    def __getitem__(self, name):
        filtered = list(filter(lambda x: x.task_name == name, self.tasks))
        assert len(filtered) < 2
        return filtered[0] if filtered else None

    def get_by_name(self, name: str) -> Task:
        filtered_tasks = list(filterfalse(lambda t: t.task_name != name, self.tasks))

        assert len(filtered_tasks) < 2

        if len(filtered_tasks) > 0:
            return filtered_tasks[0]
        return None

    @property
    def names(self):
        return [task.task_name for task in self.tasks]

    @property
    def types(self):
        return [task.task_type for task in self.tasks]

    @property
    def snapshot_dir(self):
        local_user_home_dir = f"{self.service_repo_dir}/user/{self.user_name}/"
        snapshot_path = f"{local_user_home_dir}/{self.domain_name}/{self.snapshot_dt}"
        return snapshot_path
