from abc import ABC
from pydantic import BaseModel, validator
from typing import Optional

from tunip.orjson_utils import *


class ModelConfig(BaseModel):
    model_path: str
    task_name: str
    task_type: str
    checkpoint: Optional[str] = None
    remote_host: Optional[str]
    remote_port: Optional[int]
    remote_model_name: Optional[str]
    remote_backend: Optional[str] = 'onnx'
    encoder_only: bool = False

    class Config:
        json_loads = orjson.loads
        json_dumps = orjson_dumps


class PreTrainedModelConfig(BaseModel):
    model_path: str
    model_name: str
    encoder_only: bool = False
    checkpoint: Optional[str] = None
    remote_host: Optional[str] = None
    remote_port: Optional[int] = None
    remote_model_name: Optional[str] = None
    remote_backend: Optional[str] = 'onnx'

    class Config:
        json_loads = orjson.loads
        json_dumps = orjson_dumps


class ModelOutput(ABC):
    pass


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
