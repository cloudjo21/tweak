import pathlib
import unittest
import urllib.parse

from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer
)

from tunip.env import NAUTS_LOCAL_ROOT
from tunip.path.mart import MartPretrainedModelPath
from tunip.service_config import get_service_config

from tweak.predict.models import PreTrainedModelConfig
from tweak.predict.resource_materialize import ResourceMaterializer


class ModelsTest(unittest.TestCase):
    def setUp(self):
        self.service_config = get_service_config()

    def test_load_and_forward_plm(self):
        model_name = 'monologg/koelectra-small-v3-discriminator'
        text_or_tokens = ['안녕하세요']

        model_nauts_path = MartPretrainedModelPath(
            user_name=self.service_config.username,
            model_name=urllib.parse.quote(model_name, safe='')
        )
        plm_model_path = f"{self.service_config.local_prefix}/{model_nauts_path}"

        config = AutoConfig.from_pretrained(plm_model_path)
        assert config

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_name,
            config=config
        )
        encoded = tokenizer.batch_encode_plus(
            text_or_tokens,
            max_length=128,
            padding="max_length",
            truncation=True,
            is_split_into_words=False,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )
        model = AutoModel.from_pretrained(plm_model_path, config=config)
        # model = AutoModel.from_pretrained(model_name, config=config)
        # out = model(input_ids=encoded['input_ids'])
        # print(out.last_hidden_state.size())

        out = model(input_ids=encoded["input_ids"])

        assert out
        
    def test_load_and_forward_onnx_plm(self):
        model_name = 'monologg/koelectra-small-v3-discriminator'
        text_or_tokens = ['안녕하세요']

        model_nauts_path = MartPretrainedModelPath(
            user_name=self.service_config.username,
            model_name=urllib.parse.quote(model_name, safe='')
        )
        plm_model_path = f"{self.service_config.local_prefix}/{model_nauts_path}"

        config = AutoConfig.from_pretrained(plm_model_path)
        assert config

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_name,
            config=config
        )
        encoded = tokenizer.batch_encode_plus(
            text_or_tokens,
            max_length=128,
            padding="max_length",
            truncation=True,
            is_split_into_words=False,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )
        # model = AutoModel.from_pretrained(f"{plm_model_path}/onnx/model.onnx", config=config)
        model = AutoModel.from_pretrained(f"{plm_model_path}/onnx/model.onnx", config=config)
        # model = AutoModel.from_pretrained(model_name, config=config)
        # out = model(input_ids=encoded['input_ids'])
        # print(out.last_hidden_state.size())

        out = model(input_ids=encoded["input_ids"])

        assert out
        
    def test_resource_materialize(self):
        config_json = """
            {
                "model_path": "/user/nauts/mart/plm/models/monologg%2Fkoelectra-small-v3-discriminator",
                "model_name": "monologg/koelectra-small-v3-discriminator"
            }
        """

        config = PreTrainedModelConfig.parse_raw(config_json)
        assert config

        ResourceMaterializer.apply_for_hf_model(config, self.service_config)
        model_path = f"{NAUTS_LOCAL_ROOT}/user/nauts/mart/plm/models/monologg%2Fkoelectra-small-v3-discriminator/torchscript/model.pt"
        assert pathlib.Path(model_path).exists()
