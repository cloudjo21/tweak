import pickle
import unittest

from tunip.env import NAUTS_LOCAL_ROOT
from tunip.nugget_api import Nugget
from tunip.path_utils import TaskPath
from tunip.service_config import get_service_config

from tweak.predict.builds import PredictionBuildForTokenTypeWord
from tweak.predict.models import ModelConfig
from tweak.predict.models.factory import ModelsForTokenClassificationFactory
from tweak.predict.models.hf_auto import HFAutoModelForTokenClassification
from tweak.predict.predict_token_classification import TokenClassificationPredictor
from tweak.predict.predictor import PredictorConfig
from tweak.predict.tokenizers import TokenizerConfig, TokenizersFactory


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
                "task_name": "ner",
                "max_length": 128
            }
        }
        pred_conf = PredictorConfig.parse_obj(pred_config_json)

        predictor = TokenClassificationPredictor(pred_conf)
        result = predictor.predict(["안녕하세요 저는 김철수입니다."])
        print(result)
        assert result
