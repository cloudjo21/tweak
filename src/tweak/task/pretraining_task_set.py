from enum import Enum
from typing import List, Optional
from pydantic import BaseModel

import tweak
from tweak.task import NotSupportedTaskException
from tweak.task.task_set import (
    AbstractTask,
    InputColumn,
    InputColumnType,
    TaskType
)


# class TaskType(Enum):
#     INVERSE_CLOZE_TASK = 1

#     def describe(self):
#         return self.name, self.value

#     @classmethod
#     def task_type2model_type(cls, task_type):
#         return cls.model_name2model_type(task_type.describe()[0])

#     @classmethod
#     def model_name2model_type(cls, model_name):
#         if "INVERSE_CLOZE_TASK" in model_name:
#             return tweak.model.dpr.modeling_dpr.DPRForPreTraining
#         else:
#             raise NotSupportedTaskException()


class InverseClozeTask(BaseModel, AbstractTask):
    task_name: str
    task_type: TaskType
    dataset_name: str
    dataset_path: str
    pretrained_model_name: str
    input_columns: List[InputColumn]
    label_column_name: str = None
    label_names: Optional[list] = None
    max_length: int = 128

    def get_input_column_by(
        self, column_type: InputColumnType, column_name: Optional[str] = None
    ) -> Optional[InputColumn]:
        target_input_columns = list(filter(lambda c: c.type_ == column_type, self.input_columns))
        if column_name:
            input_column: Optional[InputColumn] = list(
                filter(lambda c: c.name == column_name, target_input_columns)
            )[0]
        else:
            input_column: Optional[InputColumn] = target_input_columns[0]
        return input_column

    def has_input_column_type(self, column_type: InputColumnType) -> bool:
        input_columns: Optional[list] = (
            list(filter(lambda c: c.type_ == column_type, self.input_columns)) or None
        )
        return input_columns is not None

