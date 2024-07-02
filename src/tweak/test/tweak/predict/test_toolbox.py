import unittest

from tunip.env import NAUTS_LOCAL_ROOT
from tunip.path_utils import TaskPath
from tunip.service_config import get_service_config

from tweak.predict.predictor import PredictorConfig
from tweak.predict.toolbox import PredictionToolboxPackerForTokenClassification


class PredictorToolboxTest(unittest.TestCase):

    def setUp(self):
        service_config = get_service_config(force_service_level='dev')
        self.task_path = TaskPath(service_config.username, 'wiki_dev', '20211020_104537_425173', 'ner')

    def test_init_toolbox(self):
        pred_config_json = {
            "predict_tokenizer_type": "auto",
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
        conf = PredictorConfig.model_validate(pred_config_json)
        assert conf

        toolbox = PredictionToolboxPackerForTokenClassification.pack(conf)
        assert toolbox
