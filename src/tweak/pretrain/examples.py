import torch

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List


class PretrainingTaskType(Enum):
    NEXT_SENTENCE_PREDICTION = 0
    MASKED_LANGUAGE_MODEL = 1


class PretrainingTaskExample(ABC):
    pass


class PretrainingTaskExampleException(Exception):
    pass


class PretrainingTaskSubsequentTokensEmptyException(Exception):
    pass


@dataclass
class NextSentencePredictionExample(PretrainingTaskExample):
    tokens: List[str]
    segments: List[str]
    is_next_random: int


@dataclass 
class MaskedLanguageModelToken:
    index: int
    input_token: str
    output_label: str
    predict_mask: bool


@dataclass
class MaskedLanguageModelExample(PretrainingTaskExample):
    tokens: List[MaskedLanguageModelToken]

    @property
    def labels(self):
        return [token.output_label for token in self.tokens]
    
    @property
    def positions(self):
        return [token.index for token in self.tokens]
    
    def sort(self):
        self.tokens = sorted(self.tokens, key=lambda t: t.index)


class PretrainingModelExample(ABC):

    @abstractmethod
    def torch(self):
        pass


@dataclass
class BertModelExample(PretrainingModelExample):
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: List[int]
    is_next_random: int
    masked_lm_labels: List[int]

    def torch(self, unk_token_id):
        self.masked_lm_labels = list(map(lambda l: l if l else unk_token_id, self.masked_lm_labels))

        assert len(self.input_ids) == len(self.attention_mask)
        assert len(self.token_type_ids) == len(self.attention_mask)
        assert len(self.token_type_ids) == len(self.masked_lm_labels)

        tensors = {
            "input_ids": torch.LongTensor(self.input_ids).requires_grad_(False),
            "attention_mask": torch.LongTensor(self.attention_mask).requires_grad_(False),
            "token_type_ids": torch.LongTensor(self.token_type_ids).requires_grad_(False),
            "next_sentence_label": torch.LongTensor([self.is_next_random]).requires_grad_(False),
            "labels": torch.LongTensor(self.masked_lm_labels).requires_grad_(False)
        }
        return tensors
