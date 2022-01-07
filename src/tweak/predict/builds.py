import torch

from abc import ABC
from typing import List

from transformers.modeling_outputs import (
    TokenClassifierOutput,
)
from transformers.tokenization_utils_base import BatchEncoding

from tunip.corpus_utils import CorpusToken


class PredictionBuild(ABC):
    pass


class PredictionBuildForTokenTypeWord(PredictionBuild):

    def __call__(
        self,
        encoded: BatchEncoding,
        predictions: TokenClassifierOutput,
        label_list: List[str],
        nugget_tokens: List[CorpusToken]
    ):
        pred_result = []
        pred_label_ids = torch.argmax(predictions.logits, axis=2)
        offset_mapping = encoded["offset_mapping"]
        special_tokens_mask = encoded["special_tokens_mask"]

        normal_tokens_mask = special_tokens_mask.lt(1).bool()
        batch_ends = offset_mapping[:, :, 1]

        for i in range(len(batch_ends)):
            nonzero_ends_index = torch.where(batch_ends[i] > 0)[0]
            input_offsets = torch.index_select(
                offset_mapping[i], 0, nonzero_ends_index
            )
            word_offsets = torch.where(input_offsets[:, 0] == 0)[0]

            num_tokens = input_offsets.shape[0]
            ids = pred_label_ids[i][1 : num_tokens + 1]

            label_ids = torch.index_select(ids, 0, word_offsets)
            labels = [label_list[lid] for lid in label_ids]

            pred_result.append(list(zip(labels, nugget_tokens[i])))
        
        return pred_result
