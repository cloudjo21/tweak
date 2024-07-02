import pickle
import unittest
import urllib.parse

from tunip.path.mart import MartPretrainedModelPath, MartTokenizerPath
from tunip.service_config import get_service_config

from tweak.predict.predict_pretrained import PreTrainedModelPredictor
from tweak.predict.predictor import PredictorConfig


class PredictorsTest(unittest.TestCase):

    def setUp(self):
        self.service_config = get_service_config(force_service_level='dev')

    def test_infer(self):
        model_name='monologg/koelectra-small-v3-discriminator'

        quoted_model_name = urllib.parse.quote(model_name, safe='')
        model_nauts_path = MartPretrainedModelPath(
            user_name=self.service_config.username,
            model_name=quoted_model_name
        )
        tokenizer_nauts_path = str(model_nauts_path) + "/vocab"
        plm_model_path = str(model_nauts_path)
        tokenizer_path = str(tokenizer_nauts_path)

        pred_config_json = {
            "predict_tokenizer_type": "auto",
            "predict_model_type": "auto",
            "model_config": {
                "model_path": plm_model_path,
                "model_name": model_name
            },
            "tokenizer_config": {
                "model_path": plm_model_path,
                "path": tokenizer_path,
                "max_length": 128
            }
        }

        pred_conf = PredictorConfig.model_validate(pred_config_json)

        predictor = PreTrainedModelPredictor(pred_conf)
        result = predictor.predict(["안녕하세요 저는 김철수입니다."])
        # print(result.size())
        assert result.values

    def test_infer_for_predict_model_type_is_torchscript(self):
        model_name='monologg/koelectra-small-v3-discriminator'

        quoted_model_name = urllib.parse.quote(model_name, safe='')
        model_nauts_path = MartPretrainedModelPath(
            user_name=self.service_config.username,
            model_name=quoted_model_name
        )
        tokenizer_nauts_path = MartTokenizerPath(
            user_name=self.service_config.username,
            tokenizer_name=quoted_model_name
        )
        plm_model_path = str(model_nauts_path)
        tokenizer_path = str(tokenizer_nauts_path)

        pred_config_json = {
            "predict_tokenizer_type": "auto",
            "predict_model_type": "torchscript",
            "model_config": {
                "model_path": f"{plm_model_path}/torchscript",
                "model_name": model_name
            },
            "tokenizer_config": {
                "model_path": plm_model_path,
                "path": tokenizer_path,
                "max_length": 128
            }
        }

        pred_conf = PredictorConfig.model_validate(pred_config_json)

        predictor = PreTrainedModelPredictor(pred_conf)
        result = predictor.predict(["안녕하세요 저는 김철수입니다."])
        assert result