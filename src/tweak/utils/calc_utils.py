import math
import torch

from tunip.es_utils import init_elastic_client


def log_sum_exp(vec: torch.Tensor) -> torch.Tensor:
    """
    Calculate the log_sum_exp trick for the tensor.
    :param vec: [batch_size * from_label * to_label].
    :return: [batch_size * to_label]
    """
    max_scores, idx = torch.max(vec, 1)
    max_scores[max_scores == -float("Inf")] = 0
    # max_scores_expanded = max_scores.view(vec.shape[0], 1, vec.shape[2]).expand(
    max_scores_expanded = max_scores.reshape(vec.shape[0], 1, vec.shape[2]).expand(
        vec.shape[0], vec.shape[1], vec.shape[2]
    )
    return max_scores + torch.log(torch.sum(torch.exp(vec - max_scores_expanded), 1))


def _padded_inputs_and_masks(predictor, texts):
    inputs = predictor.tokenizer.tokenizer.batch_encode_plus(texts, padding="longest")
    # inputs = predictor.tokenizer.tokenize(texts)

    padded_input_ids = torch.LongTensor(inputs['input_ids'])
    attention_mask = torch.LongTensor(inputs['attention_mask'])

    zero_mask = torch.zeros(attention_mask.size(), dtype=torch.long)
    token_mask = torch.where(padded_input_ids == predictor.tokenizer.tokenizer.cls_token_id, zero_mask, attention_mask)
    token_mask = torch.where(padded_input_ids == predictor.tokenizer.tokenizer.sep_token_id, zero_mask, token_mask)
    
    return padded_input_ids, attention_mask, token_mask


def _create_padded_tensor(float_list, target_length):
    # Convert the list of floats to a tensor
    float_tensor = torch.tensor(float_list, dtype=torch.float32)

    # Calculate the padding length
    padding_length = target_length - len(float_list)

    # Check if padding is necessary
    if padding_length > 0:
        # Create a zero-filled tensor with the required padding length
        padding_tensor = torch.zeros(padding_length, dtype=torch.float32)

        # Concatenate the original tensor with the padding tensor
        padded_tensor = torch.cat((float_tensor, padding_tensor))
    else:
        padded_tensor = float_tensor

    return padded_tensor


def _get_idf_vector(es, text, index, field):
    body = {
        field: text
    }
    es_res = es.termvectors(index=index, doc=body, fields=[field], field_statistics=True, term_statistics=True)
    # print(es_res['term_vectors'][field])
    tokens = text.split()
    vector = []
    for t in tokens:
        if t in es_res['term_vectors'][field]['terms']:
            idf = 1./math.log(es_res['term_vectors'][field]['terms'][t]['doc_freq'])
        else:
            idf = 0.0
        vector.append(idf)
    return vector


def _compute_pairwise_cosine(refer_embeds, candi_embeds):
    """
    Args:
        refer_embeds (torch.tensor) : (B, K_i, D)
            B : batch size
            K_i : maximum sequence length in `refer_embeds`
            D : BERT embedding dim
        candi_embeds (torch.tensor) : (B, K_r, D)
            B : batch size
            K_r : maximum sequence length in `candi_embeds`
            D : BERT embedding dim

    Returns:
        pairwise_cosine (torch.tensor) : (B, K_i, K_r)

    Examples::
        >>> input1 = torch.randn(3, 4, 5)
        >>> input2 = torch.randn(3, 7, 5)
        >>> compute_pairwise_cosine(input1, input2).size()
        $ torch.Size([3, 4, 7])
    """
    def normalize(embeds):
        return torch.div(embeds, torch.norm(embeds, dim=-1).unsqueeze(-1), rounding_mode=None)
        # embeds.div_(torch.norm(embeds, dim=-1).unsqueeze(-1))
        # return embeds

    refer_embeds = normalize(refer_embeds)
    candi_embeds = normalize(candi_embeds)
    pairwise_cosine = torch.bmm(refer_embeds, candi_embeds.permute(0, 2, 1))
    return pairwise_cosine


def _get_weight_mask(texts, input_ids, nugget, es, index_name, index_field, longest_length):

    nugget_gen = nugget.record_v2(texts)
    nuggets = list(nugget_gen)
    _texts = [' '.join([t.surface for t in r.tokens]) for r in nuggets]

    # weight_mask = input_ids * torch.vstack([_create_padded_tensor(torch.tensor(_get_idf_vector(es, _text, index=index_name, field=index_field)), longest_length) for _text in _texts])
    idf_vectors = torch.vstack([_create_padded_tensor(torch.tensor(_get_idf_vector(es, _text, index=index_name, field=index_field)), longest_length) for _text in _texts])
    weight_mask = input_ids * idf_vectors
    return weight_mask


def _longest_length(refer_token_mask, candi_token_mask):
    try:
        longest_length = max(max([e.index(0, 0) for e in refer_token_mask.tolist()]), max([e.index(0, 0) for e in candi_token_mask.tolist()]))
    except:
        longest_length = refer_token_mask.shape[1]
    return longest_length


class TfidfDocumentScorer:
    def __init__(self, service_config, index_name, index_field, N, avgdl, k1=1.2, b=0.75):
        self.index_name = index_name
        self.index_field = index_field
        self.N = N
        self.avgdl = avgdl
        self.k1 = k1
        self.b = b

        self.es = init_elastic_client(service_config)

    def _get_tfidf(self, n, N, freq, dl, avgdl, k1=1.2, b=0.75):
        tf_term = freq / (freq + k1 * (1 - b + b * dl / avgdl))
        idf_term = math.log(1 + (N - n + 0.5) / (n + 0.5))
        return tf_term * idf_term

    def score(self, tokenized_text) -> dict:
        term2score = {}
        es_res = self.es.termvectors(
            index=self.index_name,
            doc={
                self.index_field: tokenized_text
            },
            fields=[self.index_field],
            field_statistics=True,
            term_statistics=True
        )

        dl = len(tokenized_text.split())
        for term, stat in es_res.body["term_vectors"]["text"]["terms"].items():
            doc_freq = stat["doc_freq"] if "doc_freq" in stat else self.N
            term2score.update({term: self._get_tfidf(n=doc_freq, N=self.N, freq=stat["term_freq"], dl=dl, avgdl=self.avgdl)})

        return term2score
