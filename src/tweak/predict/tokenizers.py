from abc import ABC, abstractmethod
from typing import Optional

from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

from tweak.orjson_utils import orjson_dumps


class TokenizerConfig(BaseModel):
    model_path: str
    task_name: str
    checkpoint: Optional[str] = None

    class Config:
        json_loads = orjson.json_loads
        json_dumps = orjson_dumps


class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, texts):
        pass


# nugget
class NuggetTokenizer(Tokenizer):
    def __init__(self, config: TokenizerConfig):
        pass


# auto tokenzier
class HFAutoTokenizer(Tokenizer):
    def __init__(self, config: TokenizerConfig):
        model_dir = f"{config.model_path}/{config.checkpoint}/{task_name}" if config.checkpoint else config.model_path

        auto_config = AutoConfig.from_pretrained(
            model_dir, finetuning_task=config.task_name
        )
        pt_model_name = auto_config._name_or_path

        self.tokenizer = AutoTokenizer.from_pretrained(
            pt_model_name,
            use_fast=True,
            add_prefix_space=False if "roberta" not in pt_model_name else True,
        )
    
    def tokenize(self, texts):
        pass


class TokenizersFactory:

    @classmethod
    def create(cls, predict_tokenizer_type: str, config: str):

        config = TokenizerConfig.parse_raw(config)

        if predict_tokenizer_type == 'nugget':
            return NuggetTokenizer(config)
        return return HFAutoTokenizer(config)
