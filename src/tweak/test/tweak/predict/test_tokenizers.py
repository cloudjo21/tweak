import pathlib
import unittest

from transformers import AutoConfig, AutoTokenizer
from transformers.file_utils import is_remote_url, get_from_cache

from tunip.env import NAUTS_LOCAL_ROOT
from tunip.path_utils import services as path_services
from tunip.service_config import get_service_config

from tweak.predict.config import TokenizerConfig
from tweak.predict.predictor import PredictorConfig
from tweak.predict.resource_materialize import ResourceMaterializer


class TokenizersTest(unittest.TestCase):
    def test_resource_materialize(self):
        service_config = get_service_config(force_service_level='dev')

        config_json = """
            {
                "model_path": "/user/nauts/mart/plm/models/monologg%2Fkoelectra-small-v3-discriminator",
                "path": "/user/nauts/mart/tokenizers/monologg%2Fkoelectra-small-v3-discriminator",
                "max_length": 128
            }
        """

        config = TokenizerConfig.parse_raw(config_json)
        assert config
        assert config.max_length == 128

        ResourceMaterializer.apply_for_tokenizer(config, service_config)
        vocab_path = f"{NAUTS_LOCAL_ROOT}/user/nauts/mart/tokenizers/monologg%2Fkoelectra-small-v3-discriminator/vocab.txt"
        assert pathlib.Path(vocab_path).exists()

    def test_load_predictor_config_and_access_config(self):

        predictor_config = {
            "predict_tokenizer_type": "auto",
            "predict_model_type": "torchscript",
            "model_config": {
                "model_path": "/user/nauts/mart/plm/models/monologg%2Fkoelectra-small-v3-discriminator/torchscript",
                "model_name": "monologg/koelectra-small-v3-discriminator"
            },
            "tokenizer_config": {
                "model_path": "/user/nauts/mart/plm/models/monologg%2Fkoelectra-small-v3-discriminator",
                "path": "/user/nauts/mart/tokenizers/monologg%2Fkoelectra-small-v3-discriminator",
                "max_length": 128
            }
        }

        pred_config = PredictorConfig.parse_obj(predictor_config)
        assert pred_config.tokenizer_config
        assert pred_config.tokenizer_config.max_length == 128