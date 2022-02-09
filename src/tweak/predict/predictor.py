import torch

from pydantic import BaseModel
from typing import Union

from tweak.predict.models import ModelConfig, PreTrainedModelConfig
from tweak.predict.tokenizers import TokenizerConfig


class Predictor(torch.nn.Module):

    def __init__(self, config):
        self.config = config 


class PredictorConfig(BaseModel):
    # nugget/auto
    predict_tokenizer_type: str
    # triton/auto
    predict_model_type: str

    model_config: Union[ModelConfig, PreTrainedModelConfig]
    tokenizer_config: TokenizerConfig
