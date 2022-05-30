from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List

from transformers import AutoConfig, AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from tunip.nugget_api import Nugget, NuggetFilterResultFormat
from tunip.service_config import get_service_config

from tweak import LOGGER
from tweak.predict.config import TokenizerConfig
from tweak.predict.resource_materialize import ResourceMaterializer


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
        self.nugget = Nugget()
        self.allow_pos_tags = config.allow_tags

    def tokenize(self, text_or_tokens) -> BatchEncoding:
        assert isinstance(text_or_tokens, list) and not isinstance(text_or_tokens[0], list)
        nugget_tokens = self.nugget(text_or_tokens)
        if self.allow_pos_tags:
            tokens = self.nugget.filter(
                nuggets, white_tags=self.allow_pos_tags, result_format=NuggetFilterResultFormat.B_E_LEX
            )
        else:
            nugget_tokens = [
                [[e[0], e[1], e[3]] for e in ent["tokens"]] for ent in result_tokens
            ]
            tokens = [[e[2] for e in ent] for ent in nugget_tokens]
        # [[start, end, surface], ...]
        return BatchEncoding(data={"nugget_tokens": [nugget_tokens]})


# auto tokenzier
class HFAutoTokenizer(Tokenizer):
    def __init__(self, config: TokenizerConfig):
        # model_dir = f"{config.model_path}/{config.checkpoint}/{config.task_name}" if config.checkpoint else config.model_path
        # model_dir = f"{config.model_path}/{config.checkpoint}/{config.task_name}" if config.checkpoint else config.model_path

        auto_config = AutoConfig.from_pretrained(
            config.model_path, # finetuning_task=config.task_name
        )
        pt_model_name = config.path if config.path else auto_config._name_or_path

        self.tokenizer = AutoTokenizer.from_pretrained(
            pt_model_name,
            config=auto_config,
            use_fast=True,
            add_prefix_space=False if "roberta" not in pt_model_name else True,
        )
        self.max_length = config.max_length
    

    def tokenize(self, text_or_tokens) -> BatchEncoding:

        is_split_into_words = (
            True if isinstance(text_or_tokens, list) and isinstance(text_or_tokens[0], list) else False
        )

        encoded = self.tokenizer.batch_encode_plus(
            text_or_tokens,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            is_split_into_words=is_split_into_words,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )
        return encoded


class NuggetHFAutoTokenizer(Tokenizer):

    def __init__(self, config: TokenizerConfig):
        self.nugget = Nugget()
        auto_config = AutoConfig.from_pretrained(
            config.model_path, # finetuning_task=config.task_name
        )
        pt_model_name = config.path if config.path else auto_config._name_or_path

        self.tokenizer = AutoTokenizer.from_pretrained(
            pt_model_name,
            config=auto_config,
            use_fast=True,
            add_prefix_space=False if "roberta" not in pt_model_name else True,
        )
        self.max_length = config.max_length
        # e.g. ["N", "V"]
        self.allow_pos_tags = config.allow_tags

    def tokenize(self, text_or_tokens) -> BatchEncoding:
        assert isinstance(text_or_tokens, list) and not isinstance(text_or_tokens[0], list)
        result_tokens = self.nugget(text_or_tokens)

        if self.allow_pos_tags:
            tokens = self.nugget.filter(
                nuggets, white_tags=self.allow_pos_tags, result_format=NuggetFilterResultFormat.B_E_LEX
            )
        else:
            nugget_tokens = [
                [[e[0], e[1], e[3]] for e in ent["tokens"]] for ent in result_tokens
            ]
            tokens = [[e[2] for e in ent] for ent in nugget_tokens]

        encoded = self.tokenizer.batch_encode_plus(
            tokens,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            is_split_into_words=True,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )
        encoded["nugget_tokens"] = nugget_tokens
        return encoded


class TokenizersFactory:

    @classmethod
    def create(cls, predict_tokenizer_type: str, config: str):

        config = TokenizerConfig.parse_raw(config)
        config = deepcopy(config)

        ResourceMaterializer.apply_for_tokenizer(config, get_service_config())

        if predict_tokenizer_type == 'nugget':
            return NuggetTokenizer(config)
        elif predict_tokenizer_type == 'nugget_auto':
            return NuggetHFAutoTokenizer(config)
        else:
            return HFAutoTokenizer(config)
