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
        text1 = "텍사스 시티"
        # text2 = "텍사스 시티 (레인져스)"
        text2 = "프랑스의 교육"
        # text2 = "OpenWrt, OpenWrt(오픈더블유알티)는 무선랜 라우터를 위한 비실시간(Non-Real-Time) 리눅스 기반의 오픈 소스 운영 체제이다. 원래는 Linksys사의 가정용 무선랜 라우터 모델인 WRT54G 시리즈의 성능을 강화하기 위한 커스텀 운영 체제로서 개발이 시작되었다가, 이후 점차 다른 무선랜 라우터들을 지원하기 시작하여 지금은 대부분의 라우터 플랫폼을 지원하고 있다. 무선랜 라우터 기능을 지원하는 임베디드 보드들은 대개 제한된 처리 능력과 메모리를 가지기 때문에 일반 PC에서와 같이 리눅스의 모든 기능을 구현하는 것이 불가능하며 라우터로서 반드시 필요한 기능들만 선택적으로 설치되어야 한다. OpenWrt는 무선랜 라우터에 필요한 리눅스의 기능들을 패키지 형태로 제공함으로써 사용자들에게 편의를 제공한다."
        # text1 = "프랑스의 교육"
        # text2 = "[CLS] 안홍준, 안홍준(安鴻俊, 1951년 3월 2일 ~ )은 대한민국의 의사이자 정치가이다.\n논란.\n야당지지자 이민 발언 논란.\n2012년 6월 25일자 경남도민일보는 안홍준 의원이 22일‘국회의원 초청 상공인 간담회’에 참석해 경남은행 분리매각 등 창원상공회의소가 마련한 13건의 건의사항을 듣고 이같은 발언을 했다고 보도하였다. 안 의원은 경남은행 관련 발언 후 대선 관련 이야기를 하면서 \"야당이 집권하면 사업하는 분들은 이민가야지 않겠느냐\"고 하면서 \"이민을 가지 않도록 해야한다\"고 주장하였다."

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
        # print(encoded1)
        # print(encoded1)
        # encoded = tokenizer.tokenize([["안녕하세요", "저", "는", "김철수", "입니다", "."]])
        # print(encoded1)
        # print(encoded2)
        assert encoded1[0].tokens[1] != UNK
        assert encoded2[0].tokens[1] != UNK

        plm_predictor = PredictorFactory.create(pred_config)
        response = plm_predictor.predict([text1, text2])
        print(response[0].shape)
        print(response[1].shape)
        print(response[0][0:10])
        print(response[1][0:10])
        # q_vector = np.mean(np.asarray(plm_res["result"]), axis=1).astype('float32')

        print(encoded1.input_ids)
        length = max((encoded1.input_ids[0] == 3).nonzero().item(), (encoded2.input_ids[0] == 3).nonzero().item())
        print(length)

        res0 = torch.mean(response[0][1:length], dim=0)
        res1 = torch.mean(response[1][1:length], dim=0)
        print(res0.shape)
        print(res1.shape)
        print(res0[0: length])
        print(res1[0: length])
        cossim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        sim = cossim(res0, res1)
        print(f"cossim: {sim}")

        assert reduce(
            lambda a, b: a == b,
            sim
        ).item() == 1.
