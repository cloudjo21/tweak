import pickle
import unittest

from tunip.env import NAUTS_LOCAL_ROOT
from tunip.nugget_api import Nugget
from tunip.path_utils import TaskPath
from tunip.service_config import get_service_config

from tweak.predict.models import ModelConfig
from tweak.predict.models.hf_auto import HFAutoModelForTokenClassification
from tweak.predict.models.factory import ModelsForTokenClassificationFactory
from tweak.predict.builds import PredictionBuildForTokenTypeWord
from tweak.predict.config import TokenizerConfig
from tweak.predict.tokenizers import TokenizersFactory


class PredictionBuildTest(unittest.TestCase):

    def setUp(self):
        service_config = get_service_config(force_service_level='dev')
        model_path = TaskPath(service_config.username, 'wiki_dev', '20211020_104537_425173', 'ner')
        self.config = ModelConfig(
            model_path=str(model_path),
            task_name="ner",
            task_type="TOKEN_CLASSIFICATION",
            checkpoint="checkpoint-55200"
        )
        self.tok_config = TokenizerConfig(
            model_path=str(model_path),
            task_name="ner",
            max_length=128
        )

        self.nugget = Nugget()
    
    def test_infer(self):
        tokenizer = TokenizersFactory.create('auto', self.tok_config.json())
        model = ModelsForTokenClassificationFactory.create('auto', self.config)
        assert model

        encoded = tokenizer.tokenize([["안녕하세요", "저", "는", "김철수", "입니다", "."]])
        out = model.infer(encoded)
        print(out.logits.shape)
        assert out.logits is not None

        label_list_path = f"{self.config.model_path}/label_list.pickle"
        with open(label_list_path, "rb") as lf:
            label_list = pickle.load(lf)

        nugget_tokens = self.nugget(["안녕하세요 저는 김철수 입니다."])
        nugget_tokens = [
            [[e[0], e[1], e[3]] for e in ent["tokens"]] for ent in nugget_tokens
        ]

        pred_result = PredictionBuildForTokenTypeWord()(encoded, out, label_list, nugget_tokens)
        assert pred_result
