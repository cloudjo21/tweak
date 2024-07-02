import os
import pathlib
import torch

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoModelForTokenClassification,
    AutoTokenizer
)

from tweak import DEFAULT_PADDING
from tweak.model.convert import DEFAULT_LANG_TO_EXAMPLES


class TorchScriptModelConverter:
    LANG_EXAMPLES = DEFAULT_LANG_TO_EXAMPLES

    def __init__(self, model_path, runtime_model_path, max_length, lang='ko', encoder_only=False):
        self.model_path = model_path
        self.runtime_model_path = runtime_model_path
        self.max_length = max_length
        self.lang = lang
        self.encoder_only = encoder_only

    def _constitute_jit_input(self, encoded):
        jit_input = []
        encoded_keys = [k for k, v in encoded.items()]
        jit_input.append(encoded['input_ids'])
        jit_input.append(encoded['attention_mask'])
        if self.encoder_only is False:
            jit_input.append(encoded['token_type_ids'])

        return encoded_keys, jit_input

    def _export(self, model, encoded):

        if self.encoder_only is True:
            self.runtime_model_path = self.runtime_model_path + '/' + 'encoder'

        pathlib.Path(self.runtime_model_path).mkdir(parents=True, exist_ok=True)

        encoded_keys, jit_input = self._constitute_jit_input(encoded)

        traced_model = torch.jit.trace(model, jit_input)
        torch.jit.save(traced_model, f"{self.runtime_model_path}/model.pt")

        print(f"input keys to torchscript model: {','.join(encoded_keys)}")


class TorchScriptHfModelConverter(TorchScriptModelConverter):
    def __init__(self, model_path, runtime_model_path, max_length, tokenizer_path_or_pt_model_name, lang, device, encoder_only=False):
        super(TorchScriptHfModelConverter, self).__init__(model_path, runtime_model_path, max_length, lang, encoder_only)
        self.tokenizer_path_or_pt_model_name = tokenizer_path_or_pt_model_name
        self.device = device

    def _make_hf_encoding_input(self, config):
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path_or_pt_model_name, config=config)
        encoded = tokenizer.batch_encode_plus(
            TorchScriptModelConverter.LANG_EXAMPLES[self.lang],
            max_length=self.max_length,
            padding=DEFAULT_PADDING,
            truncation=True,
            pad_to_max_length=True,
            return_tensors="pt"
        ).to(self.device)

        return encoded


class TorchScriptPretrainedModelConverter(TorchScriptHfModelConverter):
    def __init__(self, model_path, runtime_model_path, max_length, tokenizer_path_or_pt_model_name, lang, device, encoder_only=False):
        super(TorchScriptPretrainedModelConverter, self).__init__(model_path, runtime_model_path, max_length, tokenizer_path_or_pt_model_name, lang, device, encoder_only)
    
    def __call__(self):
        config_path = f"{self.model_path}/config.json"
        config = AutoConfig.from_pretrained(config_path, torchscript=True)

        model = AutoModel.from_pretrained(str(self.model_path), config=config).to(self.device)

        if self.encoder_only is True:
            model = model.get_encoder()

        model.eval()

        with torch.no_grad():
            encoded = self._make_hf_encoding_input(config)
            ret_code = self._export(model, encoded)

            return ret_code

class TorchScriptModelForTokenClassificationModelConverter(TorchScriptHfModelConverter):
    def __init__(self, model_path, runtime_model_path, max_length, tokenizer_path_or_pt_model_name, lang, device):
        super(TorchScriptModelForTokenClassificationModelConverter, self).__init__(model_path, runtime_model_path, max_length, tokenizer_path_or_pt_model_name, lang, device)

    def __call__(self):
        config_path = f"{self.model_path}/config.json"
        config = AutoConfig.from_pretrained(config_path, torchscript=True)
        model = AutoModelForTokenClassification.from_pretrained(str(self.model_path), config=config).to(self.device)
        
        model.eval()

        with torch.no_grad():
            encoded = self._make_hf_encoding_input(config)
            ret_code = self._export(model, encoded)

            return ret_code


class TorchScriptModelForSeq2SeqLMModelConverter(TorchScriptHfModelConverter):
    def __init__(self, model_path, runtime_model_path, max_length, tokenizer_path_or_pt_model_name, lang, device, encoder_only=False):
        super(TorchScriptModelForSeq2SeqLMModelConverter, self).__init__(model_path, runtime_model_path, max_length, tokenizer_path_or_pt_model_name, lang, device, encoder_only)

    def _constitute_jit_input(self, encoded):
        jit_input = []
        encoded_keys = [k for k, v in encoded.items()]
        jit_input.append(encoded['input_ids'])
        jit_input.append(encoded['attention_mask'])

        return encoded_keys, jit_input

    def __call__(self):
        config_path = f"{self.model_path}/config.json"
        config = AutoConfig.from_pretrained(config_path, torchscript=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(str(self.model_path), config=config).to(self.device)

        if self.encoder_only is True:
            model = model.get_encoder()
        
        model.eval()

        with torch.no_grad():
            encoded = self._make_hf_encoding_input(config)
            ret_code = self._export(model, encoded)

            return ret_code
