import pickle
import unittest
import urllib.parse

from tunip.env import NAUTS_LOCAL_ROOT
from tunip.nugget_api import Nugget
from tunip.path_utils import TaskPath
from tunip.service_config import get_service_config
from tunip.path.mart import MartPretrainedModelPath

from tweak.predict.builds import PredictionBuildForLastHiddenStateWithAttentionMaskForMeanPooling
from tweak.predict.config import TokenizerConfig
from tweak.predict.models import PreTrainedModelConfig
from tweak.predict.models.factory import ModelsForPreTrainedModelFactory
from tweak.predict.tokenizers import TokenizersFactory
from tweak.predict.predictors import PredictorFactory
from tweak.predict.predictor import PredictorConfig


class PredictionBuildPoolingTest(unittest.TestCase):

    def setUp(self):
        service_config = get_service_config(force_service_level='dev')

        self.user_name = service_config.username
        model_name='monologg/koelectra-small-v3-discriminator'
        self.quoted_model_name = urllib.parse.quote(model_name, safe='')

        self.model_path = f"{str(MartPretrainedModelPath(service_config.username, self.quoted_model_name))}"
        vocab_path = f"/user/{self.user_name}/mart/plm/models/{self.quoted_model_name}/vocab"

        model_config = {"model_path": str(self.model_path), "model_name": model_name}
        tokenizer_config = {
            "model_path": str(self.model_path),
            "path": vocab_path,
            "max_length": 128
        }

        self.model_config = PreTrainedModelConfig.model_validate(model_config)
        self.tokenizer_config = TokenizerConfig.model_validate(tokenizer_config)
        self.nugget = Nugget()
    
    def test_infer(self):
        tokenizer = TokenizersFactory.create('auto', self.tokenizer_config.json())
        model = ModelsForPreTrainedModelFactory.create('auto', self.model_config)
        assert model

        encoded = tokenizer.tokenize([["안녕하세요", "저", "는", "김철수", "입니다", "."], ["안녕하세요", "저", "는", "김철수", "입니다", "."]])
        out = model.infer(encoded)

        pred_result = PredictionBuildForLastHiddenStateWithAttentionMaskForMeanPooling()(encoded, out)
        # dim0 size must be 1 because we set the mean pooling output for plm
        assert pred_result.shape[0] == 1

    def test_infer_with_factory(self):
        pred_config_json = {
            "predict_tokenizer_type": "auto",
            "predict_model_type": "auto",
            "predict_output_type": "last_hidden_with_attention_mask.mean_pooling",
            "model_config": {
                "model_path": f"{str(self.model_path)}",
                "model_name": str(self.quoted_model_name)
            },
            "tokenizer_config": {
                "model_path": str(self.model_path),
                "path": f"/user/{self.user_name}/mart/tokenizers/{self.quoted_model_name}",
                "max_length": 128
            }
        }
        predictor = PredictorFactory.create(PredictorConfig.model_validate(pred_config_json))
        output = predictor.predict(["아이들 눈높이에서 아이들을 보시는 친절한 선생님이세요.  동화구연 으로 책읽기 놀이 돌봄수업, 한글 수업을 추천해요. "])
        assert output.shape[0] == 1
