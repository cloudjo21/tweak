from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import List, Optional

from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from tweak.orjson_utils import *


class TokenizerConfig(BaseModel):
    model_path: str
    task_name: str
    checkpoint: Optional[str] = None
    max_length: int = 128

    class Config:
        json_loads = orjson.loads
        json_dumps = orjson_dumps


class TokenizerOutput(ABC):
    pass


class NuggetTokenizerOutput(TokenizerOutput):
    tokens: List[str]


class HfAutoTokenizerOutput(TokenizerOutput):
    encoded: BatchEncoding


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
        self.max_length = config.max_length
    
    def tokenize(self, text_or_tokens):
        encoded = self.tokenizer.batch_encode_plus(
            text_or_tokens,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            is_split_into_words=True,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )
        encoded['input_ids']
        return encoded


class TokenizersFactory:

    @classmethod
    def create(cls, predict_tokenizer_type: str, config: str):

        config = TokenizerConfig.parse_raw(config)

        if predict_tokenizer_type == 'nugget':
            return NuggetTokenizer(config)
        return HFAutoTokenizer(config)
