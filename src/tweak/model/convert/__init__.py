import pathlib
import torch

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForTokenClassification,
    AutoTokenizer
)

def prepare_device(n_gpu_req):
    n_gpu = torch.cuda.device_count()
    if n_gpu_req > 0 and n_gpu == 0:
        print("There is NO GPU AVAILABLE on this machine.")
    if n_gpu_req > n_gpu:
        print(f"There is only {n_gpu} number of gpu is available, but {n_gpu_req} are requested.")
    
    device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
    device_ids = list(range(n_gpu))

    return device, device_ids


DEFAULT_PT_MODEL_NAME = {
    'ko': 'monologg/koelectra-base-v3-discriminator',
    'en': 'bert-base-cased'
}


DEFAULT_LANG_TO_EXAMPLES = {
    'ko': [
        "[CLS] 한국어 ELECTRA를 공유합니다. 한국어 ELECTRA를 공유합니다. 한국어 ELECTRA를 공유합니다. 한국어 ELECTRA를 공유합니다. 한국어 ELECTRA를 공유합니다. 한국어 ELECTRA를 공유합니다. 한국어 ELECTRA를 공유합니다. [SEP]",
        "[CLS] 한국어 ELECTRA를 공유합니다. [SEP]",
        "[CLS] 한국어 ELECTRA를 공유합니다. [SEP]",
        "[CLS] 한국어 ELECTRA를 공유합니다. 한국어 ELECTRA를 공유합니다. 한국어 ELECTRA를 공유합니다. 한국어 ELECTRA를 공유합니다. 한국어 ELECTRA를 공유합니다. 한국어 ELECTRA를 공유합니다. 한국어 ELECTRA를 공유합니다. [SEP]",
    ],

    # TODO support examples for other languages
    # 'en': [
    #     # TODO
    # ],
}


class TorchModelConverter:
    """
    torch2onnx model converter
    """

    LANG_EXAMPLES = DEFAULT_LANG_TO_EXAMPLES

    def __init__(self, model_path, runtime_model_path, max_length, lang='ko'):
        self.model_path = model_path
        self.runtime_model_path = runtime_model_path
        self.max_length = max_length
        self.lang = lang
    
    def _export(self, model, encoded):

        pathlib.Path(self.runtime_model_path).mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            model=model,
            args=tuple([v for v in encoded.values()]),
            f=f'{self.runtime_model_path}/model.onnx',
            export_params=True,
            do_constant_folding=False,
            opset_version=10,
            input_names=['input_ids', 'attention_mask', 'input'],
            output_names=['output_logits'],
            dynamic_axes={
                'input_ids': {0:'batch_size'},
                'attention_mask': {0:'batch_size'},
                'input': {0:'batch_size'},
                'output_logits': {0:'batch_size', 1: 'token_size'}
            },
            verbose=True
        )
        return 0


class TorchHfModelConverter(TorchModelConverter):
    def __init__(self, model_path, runtime_model_path, max_length, tokenizer_path_or_pt_model_name, lang):
        super(TorchHfModelConverter, self).__init__(model_path, runtime_model_path, max_length, lang)
        self.tokenizer_path_or_pt_model_name = tokenizer_path_or_pt_model_name

        # TODO n_gpu_req option
        self.device, _ = prepare_device(n_gpu_req=-1)

    def _make_hf_encoding_input(self, config):
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.tokenizer_path_or_pt_model_name,
            config=config
        )
        encoded = tokenizer.batch_encode_plus(
            TorchModelConverter.LANG_EXAMPLES[self.lang],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            pad_to_max_length=True,
            return_tensors="pt"
        ).to(self.device)

        return encoded


class TorchPretrainedModelConverter(TorchHfModelConverter):
    def __init__(self, model_path, runtime_model_path, max_length, tokenizer_path_or_pt_model_name, lang):
        super(TorchPretrainedModelConverter, self).__init__(model_path, runtime_model_path, max_length, tokenizer_path_or_pt_model_name, lang)
    
    def __call__(self):
        config_path = f"{self.model_path}/config.json"
        # config = AutoConfig.from_pretrained(self.model_path, torchscript=False)
        config = AutoConfig.from_pretrained(config_path, torchscript=False)

        model = AutoModel.from_pretrained(str(self.model_path), config=config)

        model.eval()
        model.to(self.device)

        encoded = self._make_hf_encoding_input(config)
        ret_code = self._export(model, encoded)

        return ret_code


class TorchModelForTokenClassificationModelConverter(TorchHfModelConverter):
    def __init__(self, model_path, runtime_model_path, max_length, tokenizer_path_or_pt_model_name, lang):
        super(TorchModelForTokenClassificationModelConverter, self).__init__(model_path, runtime_model_path, max_length, tokenizer_path_or_pt_model_name, lang)

    def __call__(self):
        config_path = f"{self.model_path}/config.json"
        config = AutoConfig.from_pretrained(self.model_path, torchscript=False)
        # config = AutoConfig.from_pretrained(config_path, torchscript=False)

        model = AutoModelForTokenClassification.from_pretrained(
            str(self.model_path),
            config=config
        )

        model.eval()
        model.to(self.device)

        encoded = self._make_hf_encoding_input(config)
        ret_code = self._export(model, encoded)

        return ret_code
