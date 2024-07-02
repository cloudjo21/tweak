import unittest
import json

from tunip.service_config import get_service_config
from tweak.predict.models import PreTrainedModelConfig
from tweak.predict.models.factory import ModelsForPreTrainedModelFactory
from tweak.predict.tokenizers import TokenizersFactory


class BmInferenceTest(unittest.TestCase):

    def setUp(self):
        service_config = get_service_config(force_service_level='dev')

    def test_switch_dimension(self):

        text1 = "자유민주주의(自由民主主義) 또는 서구식 민주주의(Western democracy)"

        torchscript_model_config = {"model_path": "/user/nauts/mart/plm/models/monologg%2Fkoelectra-small-v3-discriminator/torchscript", "model_name": "monologg/koelectra-small-v3-discriminator"}
        tokenizer_config = {
            "model_path": "/user/nauts/mart/plm/models/monologg%2Fkoelectra-small-v3-discriminator",
            "path": "/user/nauts/mart/plm/models/monologg%2Fkoelectra-small-v3-discriminator/vocab",
            "max_length": 128
        }

        torchscript_model_config_obj = PreTrainedModelConfig.parse_obj(torchscript_model_config)

        # plm_model = ModelsForPreTrainedModelFactory.create('auto', model_config_obj)
        torchscript_plm_model = ModelsForPreTrainedModelFactory.create('torchscript', torchscript_model_config_obj)

        tokenizer = TokenizersFactory.create('auto', json.dumps(tokenizer_config))
        tokenized = tokenizer.tokenize([text1])

        print(tokenized)

        out_torchscript_plm_model = torchscript_plm_model.infer(tokenized)
        print(out_torchscript_plm_model)

        # TODO (100, 128, 768)
        # TODO (100, 10, 768)
