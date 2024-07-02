import numpy as np
import torch

from abc import ABC
from typing import List

from transformers.modeling_outputs import (
    TokenClassifierOutput,
)
from transformers.tokenization_utils_base import BatchEncoding

from tunip.corpus_utils import CorpusToken

from tweak.utils.transformers_utils import index_fill_with_zero_vector
from tweak.predict.tokenizers import Tokenizer


class PredictionBuild(ABC):
    pass


class PredictionBuildForLastHiddenState(PredictionBuild):

    def __call__(self, encoded, predictions):
        return predictions.last_hidden_state


class PredictionBuildForLastHiddenStateWithAttentionMask(PredictionBuild):

    def __call__(self, encoded, predictions):
        return predictions.last_hidden_state, encoded['attention_mask']


class PredictionBuildForHiddenStates(PredictionBuild):

    def __call__(self, encoded, predictions):
        return predictions.encoder_hidden_states


class PredictionBuildForTorchScriptLastHiddenState(PredictionBuild):

    def __call__(self, encoded, predictions):
        return torch.tensor(predictions[0])


class PredictionBuildForTritonLastHiddenState(PredictionBuild):
    def __call__(self, encoded, predictions):
        return torch.tensor(predictions)


class PredictionBuildForTritonLastHiddenStateForMeanPooling(PredictionBuild):
    def __call__(self, encoded, predictions):
        # predictions.logits == logits__0
        length = max([e.index(1, 1) for e in encoded['special_tokens_mask'].tolist()])
        logits = predictions[:, :length]
        attention_mask_np = encoded['attention_mask'].numpy()[:, :length]
        input_mask_expanded = np.broadcast_to(np.expand_dims(attention_mask_np, axis=-1), logits.shape).astype(np.float32)
        embeddings = np.mean(logits * input_mask_expanded, axis=1)
        return torch.tensor(embeddings)


class PredictionBuildForTorchScriptLastHiddenStateForMeanPooling(PredictionBuild):

    def __call__(self, encoded, predictions):
        input_mask_expanded = encoded['attention_mask'].unsqueeze(-1).expand(predictions[0].size()).float()
        embeddings = torch.sum(predictions[0] * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return embeddings


class PredictionBuildForTritonLastHiddenStateForGlobalMeanPooling(PredictionBuild):
    def __call__(self, encoded, predictions):
        logits = predictions
        # predictions.logits == logits__0
        attention_mask_np = encoded['attention_mask'].numpy()
        input_mask_expanded = np.broadcast_to(np.expand_dims(attention_mask_np, axis=-1), logits.shape).astype(np.float32)
        embeddings = np.mean(logits * input_mask_expanded, axis=1)
        return torch.tensor(embeddings).mean(dim=0).unsqueeze(0)


class PredictionBuildForTorchScriptLastHiddenStateForGlobalMeanPool(PredictionBuild):
    def __call__(self, encoded, predictions):
        input_mask_expanded = encoded['attention_mask'].unsqueeze(-1).expand(predictions[0].size()).float()
        embeddings = torch.sum(predictions[0] * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return embeddings.mean(dim=0).unsqueeze(0)


class PredictionBuildForLastHiddenStateWithAttentionMaskForMeanPooling(PredictionBuild):
    def __call__(self, encoded, predictions):
        input_mask_expanded = encoded['attention_mask'].unsqueeze(-1).expand(predictions.last_hidden_state.size()).float()
        embeddings = torch.sum(predictions.last_hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        # TODO return attention_mask
        return embeddings.mean(dim=0).unsqueeze(0)


class PredictionBuildForTorchScriptLastHiddenStateWithZero(PredictionBuild):

    def __call__(self, encoded, predictions):
        return index_fill_with_zero_vector(encoded.input_ids, predictions[0], predictions[0].device)


class PredictionBuildForTokenGeneration(PredictionBuild):
    pass
    # def __call__(self,
    #     tokenizer: Tokenizer,
    #     model: object,
    #     encoded: BatchEncoding,
    #     predictions: np.ndarray,
    #     # label_list: List[str],
    #     nugget_tokens: List[CorpusToken],
    #     # vocab: dict
    # ):
    #     pass

class PredictionBuildForTokenTypeWord(PredictionBuild):
    def _predict(self, predictions, encoded, label_list, nugget_tokens, vocab):
        pred_result = []
        pred_label_ids = torch.argmax(predictions, axis=2)
        offset_mapping = encoded["offset_mapping"]
        special_tokens_mask = encoded["special_tokens_mask"]
        input_ids = encoded["input_ids"]

        normal_tokens_mask = special_tokens_mask.lt(1).bool()
        batch_ends = offset_mapping[:, :, 1]

        if nugget_tokens:
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
        else:
            for i in range(len(batch_ends)):
                num_tokens = torch.where(batch_ends[i] > 0)[0].shape[0]
                ids = pred_label_ids[i][1 : num_tokens + 1]

                labels = [label_list[lid] for lid in ids]

                # pred_result.append(list(zip(labels, input_ids[i][1 : num_tokens + 1])))
                pred_result.append(list(zip(labels, [vocab[tid.item()] for tid in input_ids[i][1 : num_tokens + 1]])))
        return pred_result


class PredictionBuildForHuggingFaceTokenTypeWord(PredictionBuildForTokenTypeWord):
    def __init__(self):
        super(PredictionBuildForTokenTypeWord, self).__init__()

    def __call__(
        self,
        encoded: BatchEncoding,
        predictions: TokenClassifierOutput,
        label_list: List[str],
        nugget_tokens: List[CorpusToken],
        vocab: dict
    ):
        return self._predict(predictions.logits, encoded, label_list, nugget_tokens, vocab)


class PredictionBuildForTorchScriptTokenTypeWord(PredictionBuildForTokenTypeWord):
    def __init__(self):
        super(PredictionBuildForTokenTypeWord, self).__init__()
    
    def __call__(
        self,
        encoded: BatchEncoding,
        predictions: TokenClassifierOutput,
        label_list: List[str],
        nugget_tokens: List[CorpusToken],
        vocab: dict
    ):
        return self._predict(predictions[0], encoded, label_list, nugget_tokens, vocab)


class PredictionBuildForTritonTokenTypeWord(PredictionBuildForTokenTypeWord):
    def __init__(self):
        super(PredictionBuildForTokenTypeWord, self).__init__()
    
    def __call__(
        self,
        encoded: BatchEncoding,
        predictions: np.ndarray,
        label_list: List[str],
        nugget_tokens: List[CorpusToken],
        vocab: dict
    ):
        return self._predict(torch.tensor(predictions), encoded, label_list, nugget_tokens, vocab)


class PredictionBuildForHuggingfaceTokenGen(PredictionBuildForTokenGeneration):
    def __init__(self):
        super(PredictionBuildForHuggingfaceTokenGen, self).__init__()

    def __call__(
        self,
        model: object,
        tokenizer: Tokenizer,
        inputs: BatchEncoding,
    ):
        generation_output = model.model.generate(**inputs, do_sample=True, num_beams=4, num_return_sequences=4, max_new_tokens=64)
        decode_output = tokenizer.tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        return decode_output


class PredictionBuildForTorchScriptTokenGen(PredictionBuildForTokenGeneration):
    def __init__(self):
        super(PredictionBuildForTorchScriptTokenGen, self).__init__()

    def __call__(
        self,
        model: object,
        tokenizer: Tokenizer,
        inputs: BatchEncoding,
    ):
        generation_output = model.model.generate(**inputs, do_sample=True, num_beams=4, num_return_sequences=4, max_new_tokens=64)
        decode_output = tokenizer.tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        return decode_output
