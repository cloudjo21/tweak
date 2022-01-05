import unittest

from tunip.env import NAUTS_LOCAL_ROOT
from tunip.path_utils import ModelPath
from tunip.service_config import get_service_config

from tweak.predict.models import ModelConfig
from tweak.predict.models.hf_auto import HFAutoModelForTokenClassification
from tweak.predict.models.factory import ModelsForTokenClassificationFactory
from tweak.predict.tokenizers import TokenizerConfig
from tweak.predict.tokenizers import TokenizersFactory


class ModelsForTokenClassificationTest(unittest.TestCase):

    def setUp(self):
        service_config = get_service_config(force_service_level='dev')
        model_path = ModelPath(service_config.username, 'wiki_dev', '20211020_104537_425173', 'ner', 'checkpoint-55200')
        self.config = ModelConfig(
            model_path=f"{NAUTS_LOCAL_ROOT}/{model_path}",
            task_name=''
        )
        self.tok_config = TokenizerConfig(
            model_path=f"{NAUTS_LOCAL_ROOT}/{model_path}",
            task_name='',
            max_length=128
        )

    def test_infer(self):


        tokenizer = TokenizersFactory.create('auto', self.tok_config.json())
        assert tokenizer

        model = ModelsForTokenClassificationFactory.create('auto', self.config.json())
        assert model

        model.infer()
