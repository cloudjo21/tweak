from abc import ABC
from pydantic import BaseModel
from typing import List

from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

from tweak.predict.models.hf_auto import *
from tweak.predict.models.triton import *


class ModelConfig(BaseModel):
    model_path: str
    task_name: str
    checkpoint: Optional[str] = None

    class Config:
        json_loads = orjson.json_loads
        json_dumps = orjson_dumps


class ModelOutput(BaseModel):
    logits: List[int]


class PredictableModel(ABC):
    pass


class ModelsForTokenClassificationFactory:
    @classmethod
    def create(cls, predict_model_type=str, config: str):
        if predict_model_type == 'triton':
            return TritonClientModelForTokenClassification()
        
        return HFAutoModelForTokenClassification()
