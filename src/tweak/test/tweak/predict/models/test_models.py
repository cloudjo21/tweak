import pathlib
import unittest

from tunip.env import NAUTS_LOCAL_ROOT
from tunip.path_utils import TaskPath
from tunip.service_config import get_service_config

from tweak.predict.config import TokenizerConfig
from tweak.predict.models import ModelConfig
from tweak.predict.models.factory import ModelsForTokenClassificationFactory
from tweak.predict.models.hf_auto import HFAutoModelForTokenClassification
from tweak.predict.resource_materialize import ResourceMaterializer
from tweak.predict.tokenizers import TokenizersFactory


class ModelsForTokenClassificationTest(unittest.TestCase):

    def setUp(self):
        self.service_config = get_service_config(force_service_level='dev')
        model_path = TaskPath(self.service_config.username, 'wiki_dev', '20211020_104537_425173', 'ner')
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

    def test_resource_materialize(self):
        config_json = """
            {
                "model_path": "/user/nauts/domains/wiki_dev/20211020_104537_425173/model/checkpoint-55200/ner",
                "task_name": "ner",
                "task_type": "TOKEN_CLASSIFICATION",
                "checkpoint": "checkpoint-55200"
            }
        """

        config = ModelConfig.parse_raw(config_json)
        assert config

        ResourceMaterializer.apply_for_hf_model(config, self.service_config)
        model_path = f"{NAUTS_LOCAL_ROOT}/user/nauts/domains/wiki_dev/20211020_104537_425173/model/checkpoint-55200/ner/pytorch_model.bin"
        assert pathlib.Path(model_path).exists()

    def test_download_model_file(self):
        from tunip.file_utils import HttpBasedWebHdfsFileHandler
        webhdfs = HttpBasedWebHdfsFileHandler(self.service_config)
        webhdfs.download_file(
            '/user/nauts/domains/wiki_dev/20211020_104537_425173/model/checkpoint-55200/ner/pytorch_model.bin',
            '/user/nauts/domains/wiki_dev/20211020_104537_425173/model/checkpoint-55200/ner/pytorch_model.bin',
            overwrite=True,
            read_mode='rb',
            write_mode='wb'
        )
