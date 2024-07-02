import unittest
import numpy as np

from tunip.es_utils import init_elastic_client
from tunip.nugget_api import Nugget
from tunip.service_config import get_service_config

from tweak.metrics.bert_score import calc_bert_score
from tweak.predict.predictors import SimplePredictorFactory


class BertScoreTest(unittest.TestCase):

    def test_init_predictor(self):
        model_name = "cosmoquester/bart-ko-mini"
        predictor = SimplePredictorFactory.create(model_name=model_name, plm=True, encoder_only=True, max_length=128, zero_padding=False, predict_output_type="last_hidden", device="cpu")
        assert predictor != None

    def test_bert_score(self):
        service_config = get_service_config()
        es = init_elastic_client(service_config)
        nugget = Nugget(split_sentence=True)


        index_name = "query2item_vectors"
        index_field = "text"
        model_name = "cosmoquester/bart-ko-mini"
        predictor = SimplePredictorFactory.create(model_name=model_name, plm=True, encoder_only=True, max_length=128, zero_padding=False, predict_output_type="last_hidden", device="cuda")


        references = [
            "편안하게 놀아주면서 가베 해주실 선생님 찾아요",
            "영어 파닉스와 간단한 리딩이나 동화책 읽기 편하게 해주실 선생님 찾아요",
            "수학 블록 사고력 키우기 수업 찾아요",
            "간단한 줄넘기나 자전거 타면서 아이들이랑 놀아줄 분이면 되요",
        ]
        candidates = [
            "아이와 여유롭게 가베 함께 하실 분 모셔요",
            "영어 리딩, 파닉스, 영어 놀이 영어 동화책 베이스로 기초 리딩 및 파닉스 다지기",
            "블록놀이 잘하는 레고 자란다 선생님",
            "아침 일찍 발레 교습 원해요",
            # "줄넘기 , 자전거 줄넘기 자전거 인라인",
        ]
        scores = calc_bert_score(references, candidates, predictor, nugget, es, index_name, index_field)

        assert len(list(filter(lambda x: np.isnan(x), scores.tolist()))) == 0
