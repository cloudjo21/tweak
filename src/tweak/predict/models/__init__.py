from abc import ABC
from pydantic import BaseModel
from typing import List, Optional

from tweak.orjson_utils import *


class ModelConfig(BaseModel):
    model_path: str
    task_name: str
    checkpoint: Optional[str] = None

    class Config:
        json_loads = orjson.loads
        json_dumps = orjson_dumps


class ModelOutput(BaseModel):
    logits: List[int]


class PredictableModel(ABC):
    pass
