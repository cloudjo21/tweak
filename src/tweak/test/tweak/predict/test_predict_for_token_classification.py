import unittest
import urllib.parse

from tunip.path.mart import MartPretrainedModelPath
from tunip.path_utils import TaskPath
from tunip.service_config import get_service_config

from tweak.predict.predict_token_classification import TokenClassificationPredictor
from tweak.predict.predictor import PredictorConfig
from tweak.predict.predictors import PredictorFactory
from tweak.predict.tokenizers import TokenizersFactory


class PredictorForTokenClassificationTest(unittest.TestCase):

    def setUp(self):
        service_config = get_service_config()
        task_path = TaskPath(service_config.username, 'item_description', '20221223_143738_452012', 'ner')

        model_name = 'monologg/koelectra-small-v3-discriminator'
        quoted_model_name = urllib.parse.quote(model_name, safe='')
        plm_model_path = MartPretrainedModelPath(
            user_name=service_config.username,
            model_name=quoted_model_name
        )
        tokenizer_path = str(plm_model_path) + "/vocab"
        pred_config_json = {
            "predict_tokenizer_type": "auto",
            "predict_model_type": "triton",
            # "predict_output_type": "last_hidden.mean_pooling",
            "model_config": {
                "model_path": str(task_path),
                "model_name": model_name,
                "task_name": "ner",
                "task_type": "TOKEN_CLASSIFICATION",
                "remote_host": "127.0.0.1",
                "remote_port": 31016,
                "remote_model_name": "ner"
            },
            "tokenizer_config": {
                "model_path": str(plm_model_path),
                "path": tokenizer_path,
                "max_length": 128
            }
        }
        self.pred_config = PredictorConfig.model_validate(pred_config_json)

    def test_predict_for_triton(self):
        predictor = PredictorFactory.create(self.pred_config)

        print(self.pred_config)

        res = predictor.predict(['영어 되시는 베이비시터 만 20세 이상 구합니다.'])
        assert res is not None
