import json
import torch

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

    def _get_batch_shape(self, batch_encoding):
        if 'input_ids' in batch_encoding:
            batch_shape = batch_encoding['input_ids'].shape
        elif 'inputs_embeds' in batch_encoding:
            batch_shape = batch_encoding['inputs_embeds'].shape
        else:
            raise Exception(f'CANNOT INFER token_type_ids from {batch_encoding.keys()}')
        return batch_shape

    def enforce_token_type_ids(self, batch_encoding: BatchEncoding):
        if 'token_type_ids' not in batch_encoding and self.force_token_type_ids:
            batch_shape = self._get_batch_shape(batch_encoding) 
            batch_encoding['token_type_ids'] = torch.zeros(batch_shape)
        return batch_encoding


# nugget
class NuggetTokenizer(Tokenizer):
    def __init__(self, config: TokenizerConfig):
        self.nugget = Nugget()
        self.allow_pos_tags = config.allow_tags
        self.force_token_type_ids = config.force_token_type_ids

    def tokenize(self, text_or_tokens) -> BatchEncoding:
        assert isinstance(text_or_tokens, list) and not isinstance(text_or_tokens[0], list)
        nugget_tokens = self.nugget(text_or_tokens)
        if self.allow_pos_tags:
            result_tokens = self.nugget.filter(
                nugget_tokens, white_tags=self.allow_pos_tags, result_format=NuggetFilterResultFormat.B_E_LEX
            )
        else:
            result_tokens = [
                [[e[0], e[1], e[3]] for e in ent["tokens"]] for ent in nugget_tokens
            ]
        batch_encoding = BatchEncoding(data={"nugget_tokens": [result_tokens]})
        return self.enforce_token_type_ids(batch_encoding)


# auto tokenzier
class HFAutoTokenizer(Tokenizer):
    def __init__(self, config: TokenizerConfig):
        # model_dir = f"{config.model_path}/{config.checkpoint}/{config.task_name}" if config.checkpoint else config.model_path

        auto_config = AutoConfig.from_pretrained(
            config.model_path, # finetuning_task=config.task_name
        )
        pt_model_name = config.path if config.path else auto_config._name_or_path

        try:
            self.padding = config.padding
            self.tokenizer = AutoTokenizer.from_pretrained(
                pt_model_name,
                config=auto_config,
                use_fast=True,
                local_files_only=True,
                add_prefix_space=False if "roberta" not in pt_model_name else True,
            )
        except json.decoder.JSONDecodeError as jde:
            LOGGER.error(f"{pt_model_name}")
            LOGGER.error(f"Config for Tokenizer: {auto_config}")
            # try again
            self.tokenizer = AutoTokenizer.from_pretrained(
                pt_model_name,
                config=auto_config,
                use_fast=True,
                local_files_only=True,
                add_prefix_space=False if "roberta" not in pt_model_name else True,
            )
        self.max_length = config.max_length
        self.force_token_type_ids = config.force_token_type_ids
    

    def tokenize(self, text_or_tokens) -> BatchEncoding:

        is_split_into_words = (
            True if isinstance(text_or_tokens, list) and isinstance(text_or_tokens[0], list) else False
        )

        encoded = self.tokenizer.batch_encode_plus(
            text_or_tokens,
            max_length=self.max_length,
            padding=self.padding,
            truncation=True,
            is_split_into_words=is_split_into_words,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )
        return self.enforce_token_type_ids(encoded)


class NuggetHFAutoTokenizer(Tokenizer):

    def __init__(self, config: TokenizerConfig):
        self.nugget = Nugget()
        auto_config = AutoConfig.from_pretrained(
            config.model_path, # finetuning_task=config.task_name
        )
        pt_model_name = config.path if config.path else auto_config._name_or_path

        self.padding = config.padding
        self.tokenizer = AutoTokenizer.from_pretrained(
            pt_model_name,
            config=auto_config,
            use_fast=True,
            local_files_only=True,
            add_prefix_space=False if "roberta" not in pt_model_name else True,
        )
        self.max_length = config.max_length
        # e.g. ["N", "V"]
        self.allow_pos_tags = config.allow_tags
        self.force_token_type_ids = config.force_token_type_ids

    def tokenize(self, text_or_tokens) -> BatchEncoding:
        assert isinstance(text_or_tokens, list) and not isinstance(text_or_tokens[0], list)
        nugget_tokens = self.nugget(text_or_tokens)
        if self.allow_pos_tags:
            result_tokens = self.nugget.filter(
                nugget_tokens, white_tags=self.allow_pos_tags, result_format=NuggetFilterResultFormat.B_E_LEX
            )
        else:
            result_tokens = [
                [[e[0], e[1], e[3]] for e in ent["tokens"]] for ent in nugget_tokens
            ]
        tokens = [[e[2] for e in ent] for ent in result_tokens]
        encoded = self.tokenizer.batch_encode_plus(
            tokens,
            max_length=self.max_length,
            padding=self.padding,
            truncation=True,
            is_split_into_words=True,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )
        encoded["nugget_tokens"] = result_tokens
        return self.enforce_token_type_ids(encoded)


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
