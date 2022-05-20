import pathlib
import torch

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForTokenClassification,
    AutoTokenizer
)

from tweak.model.convert import DEFAULT_LANG_TO_EXAMPLES


class TorchScriptModelConverter:
    LANG_EXAMPLES = DEFAULT_LANG_TO_EXAMPLES

    def __init__(self, model_path, runtime_model_path, max_length, lang='ko'):
        self.model_path = model_path
        self.runtime_model_path = runtime_model_path
        self.max_length = max_length
        self.lang = lang

    def _export(self, model, encoded):

        pathlib.Path(self.runtime_model_path).mkdir(parents=True, exist_ok=True)

        jit_input = []
        encoded_keys = []
        for k, v in encoded.items():
            encoded_keys.append(k)
            jit_input.append(torch.tensor(v.numpy().tolist()))
        traced_model = torch.jit.trace(model, jit_input)
        torch.jit.save(traced_model, f"{self.runtime_model_path}/model.pt")

        print(f"input keys to torchscript model: {','.join(encoded_keys)}")


class TorchScriptHfModelConverter(TorchScriptModelConverter):
    def __init__(self, model_path, runtime_model_path, max_length, tokenizer_path_or_pt_model_name, lang):
        super(TorchScriptHfModelConverter, self).__init__(model_path, runtime_model_path, max_length, lang)
        self.tokenizer_path_or_pt_model_name = tokenizer_path_or_pt_model_name

    def _make_hf_encoding_input(self, config):
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path_or_pt_model_name, config=config)
        encoded = tokenizer.batch_encode_plus(
            TorchScriptModelConverter.LANG_EXAMPLES[self.lang],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            pad_to_max_length=True,
            return_tensors="pt"
        )

        return encoded


class TorchScriptPretrainedModelConverter(TorchScriptHfModelConverter):
    def __init__(self, model_path, runtime_model_path, max_length, tokenizer_path_or_pt_model_name, lang):
        super(TorchScriptPretrainedModelConverter, self).__init__(model_path, runtime_model_path, max_length, tokenizer_path_or_pt_model_name, lang)
    
    def __call__(self):
        config_path = f"{self.model_path}/config.json"
        config = AutoConfig.from_pretrained(config_path, torchscript=True)

        model = AutoModel.from_pretrained(str(self.model_path), config=config)

        model.eval()

        with torch.no_grad():
            encoded = self._make_hf_encoding_input(config)
            ret_code = self._export(model, encoded)

            return ret_code
