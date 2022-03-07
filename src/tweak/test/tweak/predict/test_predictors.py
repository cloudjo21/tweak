import pickle
import torch
import unittest

from functools import reduce

from tunip.constants import UNK
from tunip.env import NAUTS_LOCAL_ROOT
from tunip.nugget_api import Nugget
from tunip.path_utils import TaskPath
from tunip.service_config import get_service_config

from tweak.predict.predict_token_classification import TokenClassificationPredictor
from tweak.predict.predictor import PredictorConfig
from tweak.predict.predictors import PredictorFactory
from tweak.predict.tokenizers import TokenizersFactory


class PredictorsTest(unittest.TestCase):

    def setUp(self):
        service_config = get_service_config(force_service_level='dev')
        self.task_path = TaskPath(service_config.username, 'wiki_dev', '20211020_104537_425173', 'ner')


    def test_infer(self):

        pred_config_json = {
            "predict_tokenizer_type": "nugget_auto",
            "predict_model_type": "auto",
            "model_config": {
                "model_path": str(self.task_path),
                "task_name": "ner",
                "task_type": "TOKEN_CLASSIFICATION",
                "checkpoint": "checkpoint-55200"
            },
            "tokenizer_config": {
                "model_path": str(self.task_path),
                "max_length": 128
            }
        }
        pred_conf = PredictorConfig.parse_obj(pred_config_json)

        predictor = TokenClassificationPredictor(pred_conf)
        result = predictor.predict(["안녕하세요 저는 김철수입니다."])
        assert result

    def test_compare_model_and_optimized_model_for_plm(self):
        import json
        from tweak.predict.models import PreTrainedModelConfig
        from tweak.predict.models.factory import ModelsForPreTrainedModelFactory

        text1 = "자유민주주의(自由民主主義) 또는 서구식 민주주의(Western democracy)"

        model_config = {"model_path": "/user/nauts/mart/plm/models/monologg%2Fkoelectra-small-v3-discriminator.bak", "model_name": "monologg/koelectra-small-v3-discriminator"}
        torchscript_model_config = {"model_path": "/user/nauts/mart/plm/models/monologg%2Fkoelectra-small-v3-discriminator/torchscript", "model_name": "monologg/koelectra-small-v3-discriminator"}
        tokenizer_config = {
            "model_path": "/user/nauts/mart/plm/models/monologg%2Fkoelectra-small-v3-discriminator",
            "path": "/user/nauts/mart/plm/models/monologg%2Fkoelectra-small-v3-discriminator/vocab",
            "max_length": 128
        }

        model_config_obj = PreTrainedModelConfig.parse_obj(model_config)
        torchscript_model_config_obj = PreTrainedModelConfig.parse_obj(torchscript_model_config)

        plm_model = ModelsForPreTrainedModelFactory.create('auto', model_config_obj)
        torchscript_plm_model = ModelsForPreTrainedModelFactory.create('torchscript', torchscript_model_config_obj)

        tokenizer = TokenizersFactory.create('auto', json.dumps(tokenizer_config))
        tokenized = tokenizer.tokenize([text1])

        out_plm_model = plm_model.infer(tokenized)
        # print(out_plm_model.last_hidden_state.shape)
        # print(type(out_plm_model.last_hidden_state))
        # print(out_plm_model.last_hidden_state[0][:10])

        out_torchscript_plm_model = torchscript_plm_model.infer(tokenized)
        # print(out_torchscript_plm_model[0].shape)
        # print(type(out_torchscript_plm_model[0]))
        # print(out_torchscript_plm_model[0][0][:10])

        assert reduce(
            lambda a, b: a == b,
            torch.isclose(out_plm_model.last_hidden_state.flatten(), out_torchscript_plm_model[0].flatten())
        ).item() is True


    def test_compute_similarity(self):
        text1 = "자유민주주의(自由民主主義) 또는 서구식 민주주의(Western democracy)"
        text2 = "자유민주주의(自由民主主義) 또는 서구식 민주주의(Western democracy)"

        predictor_config = {
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

        pred_config = PredictorConfig.parse_obj(predictor_config)

        tokenizer = TokenizersFactory.create('auto', pred_config.tokenizer_config.json())
        encoded1 = tokenizer.tokenize([text1])
        encoded2 = tokenizer.tokenize([text2])
        # encoded = tokenizer.tokenize([["안녕하세요", "저", "는", "김철수", "입니다", "."]])
        # print(encoded1)
        # print(encoded2)
        assert encoded1[0].tokens[1] != UNK
        assert encoded2[0].tokens[1] != UNK

        plm_predictor = PredictorFactory.create(pred_config)
        response = plm_predictor.predict([text1, text2])
        # print(response[0].shape)
        # print(response[1].shape)

        cossim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        sim = cossim(response[0], response[1])

        assert reduce(
            lambda a, b: a == b,
            sim
        ).item() == 1.
