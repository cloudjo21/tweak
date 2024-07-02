import unittest
import urllib.parse

from tunip.path.mart import MartPretrainedModelPath, MartTokenizerPath
from tunip.service_config import get_service_config

from tweak.predict.predictor import PredictorConfig
from tweak.predict.predictors import PredictorFactory


class PredictPretrainedTest(unittest.TestCase):

    def setUp(self):
        self.service_config = get_service_config()

        self.model_name = 'monologg/koelectra-small-v3-discriminator'

        username = self.service_config.username

        quoted_model_name = urllib.parse.quote(self.model_name, safe='')
        self.plm_model_path = str(
            MartPretrainedModelPath(
                user_name=self.service_config.username,
                model_name=quoted_model_name
            )
        )
        self.tokenizer_path = self.plm_model_path + "/vocab"
        pred_config_json = {
            "predict_tokenizer_type": "auto",
            "predict_model_type": "auto",
            "model_config": {
                "model_path": self.plm_model_path,
                "model_name": self.model_name
            },
            "tokenizer_config": {
                "model_path": self.plm_model_path,
                "path": self.tokenizer_path,
                "max_length": 128
            }
        }

        self.pred_config = PredictorConfig.model_validate(pred_config_json)

         
    def test_predict(self):
        predictor = PredictorFactory.create(self.pred_config)

        res = predictor.predict(['보드게임'])
        assert res is not None

    def test_predict_for_triton(self):
        pred_config_json = {
            "predict_tokenizer_type": "auto",
            "predict_model_type": "triton",
            "predict_output_type": "last_hidden.mean_pooling",
            "model_config": {
                "model_path": self.plm_model_path,
                "model_name": self.model_name,
                "remote_host": "127.0.0.1",
                "remote_port": "31016",
                "remote_model_name": "plm",
            },
            "tokenizer_config": {
                "model_path": self.plm_model_path,
                "path": self.tokenizer_path,
                "max_length": 128
            }
        }

        self.pred_config = PredictorConfig.model_validate(pred_config_json)

        predictor = PredictorFactory.create(self.pred_config)

        res = predictor.predict(['영어 되시는 베이비시터 만 20세 이상 구합니다.'])
        assert res is not None

    def test_bulk_predict_for_triton(self):
        pred_config_json = {
            "predict_tokenizer_type": "auto",
            "predict_model_type": "triton",
            "predict_output_type": "last_hidden.mean_pooling",
            "model_config": {
                "model_path": self.plm_model_path,
                "model_name": self.model_name,
                "remote_host": "0.0.0.0",
                "remote_port": "31016", "remote_model_name": "plm",
            },
            "tokenizer_config": {
                "model_path": self.plm_model_path,
                "path": self.tokenizer_path,
                "max_length": 128
            }
        }

        self.pred_config = PredictorConfig.model_validate(pred_config_json)

        predictor = PredictorFactory.create(self.pred_config)

        import time

        trials = 1000

        current = time.time()
        for _ in range(0, trials):
            res = predictor.predict(
                [
                    '영어 되시는 베이비시터 만 20세 이상 구합니다. 영어 되시는 베이비시터 만 20세 이상 구합니다.',
                    '한국어 되시는 베이비시터 만 20세 이상 구합니다. 한국어 되시는 베이비시터 만 20세 이상 구합니다.',
                    '일본어 되시는 베이비시터 만 20세 이상 구합니다. 일본어 되시는 베이비시터 만 20세 이상 구합니다.',
                    '중국어 되시는 베이비시터 만 20세 이상 구합니다. 중국어 되시는 베이비시터 만 20세 이상 구합니다.',
                    '영어 되시는 베이비시터 만 20세 이상 구합니다. 영어 되시는 베이비시터 만 20세 이상 구합니다.',
                    '한국어 되시는 베이비시터 만 20세 이상 구합니다. 한국어 되시는 베이비시터 만 20세 이상 구합니다.',
                    '일본어 되시는 베이비시터 만 20세 이상 구합니다. 일본어 되시는 베이비시터 만 20세 이상 구합니다.',
                    '중국어 되시는 베이비시터 만 20세 이상 구합니다. 중국어 되시는 베이비시터 만 20세 이상 구합니다.',
                ]
            )

        duration = time.time() - current
        print(duration)
        print(duration / (trials * 8))

        assert res is not None
