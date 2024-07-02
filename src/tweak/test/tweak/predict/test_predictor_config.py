import unittest
import urllib.parse

from tunip.path.mart import MartPretrainedModelPath, MartTokenizerPath
from tunip.path_utils import TaskPath
from tunip.service_config import get_service_config

from tweak.predict.predictor import PredictorConfig


class PredictorConfigTest(unittest.TestCase):

    def setUp(self):
        self.service_config = get_service_config(force_service_level='dev')
        self.model_path = TaskPath(self.service_config.username, 'wiki_dev', '20211020_104537_425173', 'ner')

    def test_init_config(self):
        pred_config_json = {
            "predict_tokenizer_type": "auto",
            "predict_model_type": "auto",
            "model_config": {
                "model_path": str(self.model_path),
                "task_name": "ner",
                "task_type": "TOKEN_CLASSIFICATION",
                "checkpoint": "checkpoint-55200"
            },
            "tokenizer_config": {
                "model_path": str(self.model_path),
                "max_length": 128
            }
        }
        conf = PredictorConfig.model_validate(pred_config_json)
        assert conf

    def test_init_config_with_pretrained_model(self):
        model_name='monologg/koelectra-small-v3-discriminator'

        quoted_model_name = urllib.parse.quote(model_name, safe='')
        plm_model_path = MartPretrainedModelPath(
            user_name=self.service_config.username,
            model_name=quoted_model_name
        )
        tokenizer_path = MartTokenizerPath(
            user_name=self.service_config.username,
            tokenizer_name=quoted_model_name
        )
        pred_config_json = {
            "predict_tokenizer_type": "auto",
            "predict_model_type": "auto",
            "model_config": {
                "model_path": str(plm_model_path),
                "model_name": model_name
            },
            "tokenizer_config": {
                "model_path": str(plm_model_path),
                "path": str(tokenizer_path),
                "max_length": 128
            }
        }

        conf = PredictorConfig.model_validate(pred_config_json)
        assert conf
