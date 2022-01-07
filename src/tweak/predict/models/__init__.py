from abc import ABC
from pydantic import BaseModel, validator
from typing import List, Optional

from tweak.orjson_utils import *


class ModelConfig(BaseModel):
    model_path: str
    task_name: str
    checkpoint: Optional[str]

    class Config:
        json_loads = orjson.loads
        json_dumps = orjson_dumps


class ModelOutput(ABC):
    pass


import numpy as np
import torch

# TODO for prediction result by prediction build command
class ModelOutputForTokenClassification(BaseModel, ModelOutput):
    logits: list
    @validator('logits', pre=True)
    def parse_logits(v):
        return v.tolist()

    class Config:
        arbitrary_types_allowed = True


class PredictableModel(ABC):
    pass
