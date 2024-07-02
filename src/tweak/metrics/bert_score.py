import datasets
import numpy as np
import torch

from tweak.predict.models import PredictableModel
from tweak.predict.tokenizers import Tokenizer
from tweak.utils.calc_utils import (
    _compute_pairwise_cosine,
    _get_weight_mask,
    _longest_length,
    _padded_inputs_and_masks,
)


def calc_bert_score(references, candidates, predictor, nugget, es, index_name, index_field, base=0.5, device="cpu"):

    refer_input_ids, refer_attn_mask, refer_token_mask = _padded_inputs_and_masks(predictor, references)
    candi_input_ids, candi_attn_mask, candi_token_mask = _padded_inputs_and_masks(predictor, candidates)
    longest_length = _longest_length(refer_token_mask, candi_token_mask)

    refer_weight_mask = _get_weight_mask(references, refer_input_ids, nugget, es, index_name, index_field, longest_length)
    candi_weight_mask = _get_weight_mask(candidates, candi_input_ids, nugget, es, index_name, index_field, longest_length)

    refer_embeds = predictor.predict(references).to(device)
    candi_embeds = predictor.predict(candidates).to(device)

    pairwise_cosine = _compute_pairwise_cosine(refer_embeds[:, :longest_length], candi_embeds[:, :longest_length])
    P_max, _ = pairwise_cosine.max(dim=1)
    R_max, _ = pairwise_cosine.max(dim=2)
    P_max = (P_max - base) / (1 - base)
    R_max = (R_max - base) / (1 - base)

    R = (R_max * refer_weight_mask).sum(axis=1) / refer_weight_mask.sum(axis=1)
    P = (P_max * candi_weight_mask).sum(axis=1) / candi_weight_mask.sum(axis=1)

    F = 2 * (P * R) / (P + R)
    return F


class BERTScorer:
    def __init__(self, predictor, device="cpu"):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.predictor = predictor

    def get_embeddings(self, sentences):
        with torch.no_grad():
            outputs = self.predictor.predict(sentences)
        return outputs

    def score(self, sentences1, sentences2):
        embeddings1 = self.get_embeddings(sentences1)
        embeddings2 = self.get_embeddings(sentences2)

        cosine_sim = torch.nn.functional.cosine_similarity(embeddings1, embeddings2)
        return cosine_sim.item()
    
    @classmethod
    def normalized_embeddings(embeddings):
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
        return normalized_embeddings


# predictor = SimplePredictorFactory.create(
#     model_name="cosmoquester/bart-ko-mini",
#     plm=True, encoder_only=True,
#     max_length=128,
#     zero_padding=False,
#     predict_output_type="last_hidden.mean_pooling",
#     device="cuda"
# )
# scorer = BERTScorer(predictor, device="cuda")


class BertScore(datasets.Metric):
    # TODO support it by MetricComputeFactory, pass the necessary resources
    def _info(self):
        return datasets.MetricInfo(
            description="",
            citation="",
            inputs_description="",
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("int64")),
                    "references": datasets.Sequence(datasets.Value("int64"))
                }
            )
        )

    def _compute(self, label_list, predictions, references):

        return {
            "bert_score": None
        }
