import torch

from typing import Dict, List

from transformers import DataCollatorForTokenClassification, Trainer
from transformers.data.data_collator import InputDataClass
from transformers.data.data_collator import default_data_collator


class DataCollatorFactory:
    @classmethod
    def create(cls, task_name, **kwargs):
        if task_name in ["ner"]:
            return DataCollatorForTokenClassification(kwargs["tokenizer"])
        else:
            return default_data_collator


class DummyDataCollator:
    def __call__(self, features: List[InputDataClass]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Not support this methods")

    def collate_batch(self, features: List[InputDataClass]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Not support this methods")
