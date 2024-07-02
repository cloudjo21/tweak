import torch
import unittest

from tweak.predict.predictor import PredictorConfig
from tweak.predict.predictors import PredictorFactory
from tweak.predict.tokenizers import TokenizersFactory
from tweak.utils.transformers_utils import (
    length_of_input_ids,
    zero_indexes_of_input_ids,
    index_fill_with_zero_vector
)


class TransformersUtilsTest(unittest.TestCase):

    def setUp(self):
        self.text1 = "텍사스 시티"
        self.text2 = "프랑스의 교육"

        self.predictor_config = {
            "predict_tokenizer_type": "auto",
            "predict_model_type": "torchscript",
            "model_config": {
                "model_path": "/user/nauts/mart/plm/models/monologg%2Fkoelectra-small-v3-discriminator/torchscript",
                "model_name": "monologg/koelectra-small-v3-discriminator"
            },
            "tokenizer_config": {
                "model_path": "/user/nauts/mart/plm/models/monologg%2Fkoelectra-small-v3-discriminator",
                "path": "/user/nauts/mart/plm/models/monologg%2Fkoelectra-small-v3-discriminator/vocab",
                "max_length": 128
            }
        }
        self.pred_config = PredictorConfig.model_validate(self.predictor_config)
        self.tokenizer = TokenizersFactory.create('auto', self.pred_config.tokenizer_config.json())


    def test_tailing_indexes(self):
        mtx = torch.Tensor(
            [
                [[1.0, 0.0, 0.0, 0.0], [2.0, 2.0, 0.0, 0.0], [3.0, 3.0, 3.0, 0.0]],
                [[4.0, 0.0, 0.0, 0.0], [5.0, 5.0, 0.0, 0.0], [6.0, 6.0, 6.0, 0.0]],
            ]
        )
        valid_mtx_len = 2

        zero_idx = [list(range(valid_mtx_len, mtx.shape[1])) for _ in range(len(mtx))]
        # print(zero_idx)
        assert zero_idx == [[2], [2]]


    def test_length_of_input_ids(self):
        encodings = self.tokenizer.tokenize([self.text1, self.text2])

        plm_predictor = PredictorFactory.create(self.pred_config)
        response = plm_predictor.predict([self.text1, self.text2])

        lengths = length_of_input_ids(encodings.input_ids)
        assert lengths[0] == 3
        assert lengths[1] == 4


    def test_zero_indexes_of_input_ids(self): 
        encodings = self.tokenizer.tokenize([self.text1, self.text2])

        plm_predictor = PredictorFactory.create(self.pred_config)
        response = plm_predictor.predict([self.text1, self.text2])

        zero_indexes = zero_indexes_of_input_ids(encodings.input_ids)
        assert len(zero_indexes[0]) == len(encodings.input_ids[0]) - 3
        assert len(zero_indexes[1]) == len(encodings.input_ids[0]) - 4


    def test_index_fill_with_zero_vectors(self):
        encodings = self.tokenizer.tokenize([self.text1, self.text2])

        plm_predictor = PredictorFactory.create(self.pred_config)
        response = plm_predictor.predict([self.text1, self.text2])

        encoded_with_zeros = index_fill_with_zero_vector(encodings.input_ids, response)

        assert torch.equal(encoded_with_zeros[0][3], torch.zeros(response.shape[-1], dtype=torch.float32))
        assert torch.equal(encoded_with_zeros[1][4], torch.zeros(response.shape[-1], dtype=torch.float32))

        assert encoded_with_zeros.shape == (2, 128, 256)