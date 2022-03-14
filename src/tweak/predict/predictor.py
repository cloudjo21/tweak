import torch

from pydantic import BaseModel
from typing import Optional, Union

from tweak.predict.models import ModelConfig, PreTrainedModelConfig
from tweak.predict.config import TokenizerConfig


class Predictor(torch.nn.Module):

    def __init__(self, config):
        self.config = config 


class PredictorConfig(BaseModel):
    # nugget/auto
    predict_tokenizer_type: str
    # triton/auto
    predict_model_type: str

    predict_output_type: Optional[str]

    model_config: Union[ModelConfig, PreTrainedModelConfig]
    tokenizer_config: TokenizerConfig

    zero_padding: bool = True
