import unittest

from tunip.env import NAUTS_LOCAL_ROOT
from tunip.path_utils import TaskPath
from tunip.service_config import get_service_config

from tweak.predict.models import ModelConfig
from tweak.predict.models.factory import ModelsForTokenClassificationFactory
from tweak.predict.models.hf_auto import HFAutoModelForTokenClassification
from tweak.predict.tokenizers import TokenizerConfig
from tweak.predict.tokenizers import TokenizersFactory


class ModelsForTokenClassificationTest(unittest.TestCase):

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
    
    def test_create_tokenizer(self):
        tokenizer = TokenizersFactory.create('auto', self.tok_config.json())
        assert tokenizer

    def test_tokenize_of_tokenizer(self):
        tokenizer = TokenizersFactory.create('auto', self.tok_config.json())
        assert tokenizer

    def test_infer(self):
        tokenizer = TokenizersFactory.create('auto', self.tok_config.json())
        model = ModelsForTokenClassificationFactory.create('auto', self.config)
        assert model

        encoded = tokenizer.tokenize([["안녕하세요", "저", "는", "김철수", "입니다", "."]])
        out = model.infer(encoded)
        print(out.logits.shape)
        assert out.logits is not None