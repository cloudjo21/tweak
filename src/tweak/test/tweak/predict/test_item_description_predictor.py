import pickle
import torch
import unittest

from tunip.service_config import get_service_config
from tunip.path_utils import TaskPath
from tunip.path.mart import MartPretrainedModelPath, MartTokenizerPath

from tweak.predict.predictor import PredictorConfig
from tweak.predict.predict_token_classification import TokenClassificationPredictor

class ItemDescriptionPredictorTest(unittest.TestCase):
    def setUp(self):
        service_config = get_service_config(force_service_level="dev")
        self.task_path = TaskPath(service_config.username, 'item_description', '20221028_112748_352686', 'ner')
        self.plm_model_path = f"{str(MartPretrainedModelPath(user_name=service_config.username, model_name='klue%2Froberta-base'))}"
        self.plm_tokenizer_path = f"{str(MartTokenizerPath(user_name=service_config.username, tokenizer_name='klue%2Froberta-base'))}"

    def test_infer(self):
        pred_config_json = {
            "predict_tokenizer_type": "auto",
            "predict_model_type": "auto",
            "model_config": {
                "model_path": str(self.task_path),
                "task_name": "ner",
                "task_type": "TOKEN_CLASSIFICATION",
                "checkpoint": "checkpoint-1050"
            },
            "tokenizer_config": {
                "model_path": self.plm_model_path,
                "path": f"{self.plm_model_path}/vocab",
                "max_length": 512
            }
        }
        pred_conf = PredictorConfig.model_validate(pred_config_json)

        predictor = TokenClassificationPredictor(pred_conf)
        result = predictor.predict(['수학과 과학은 항상 어디 가서도 잘하는 편에 속했습니다. 수학은 강남의 초등학교에서도 우수한 편이었고 중고등학교에서도 과목 우수상과 1등급을 받아왔습니다. 과학도 마찬가지로 우수한 성적을 거두는 편이었고, 수학 과학 모두 가르치는 경험이 풍부합니다. 현재도 수학 질문 조교로 일하고 있습니다. 초등학생이었던 사촌 동생들도 저에게 수학을 배우고 나서 실력이 향상된 사례가 있습니다!'])
        assert result
        