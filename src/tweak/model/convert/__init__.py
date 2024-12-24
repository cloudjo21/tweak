import onnx
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

    def __init__(self, model_path, runtime_model_path, max_length, lang='ko', encoder_only=False):
        self.model_path = model_path
        self.runtime_model_path = runtime_model_path
        self.max_length = max_length
        self.lang = lang
        self.encoder_only = encoder_only

    def _constitute_input_name_and_dynamic_axes(self):
        input_names = ['input_ids', 'attention_mask', 'token_type_ids']
        dynamic_axes = {
            'input_ids': {0:'batch_size', 1:'sequence'},
            'attention_mask': {0:'batch_size', 1:'sequence'},
            'token_type_ids': {0:'batch_size', 1:'sequence'},
            'output_logits': {0:'batch_size', 1: 'token_size'}
        }
        return input_names, dynamic_axes

    def _constitute_input_name_and_dynamic_axes_for_encoder_only(self):
        input_names = ['input_ids', 'attention_mask']
        dynamic_axes = {
            'input_ids': {0:'batch_size', 1:'sequence'},
            'attention_mask': {0:'batch_size', 1:'sequence'},
            'output_logits': {0:'batch_size', 1: 'token_size'}
        }
        return input_names, dynamic_axes

    def _export(self, model, encoded):

        if self.encoder_only is True:
            input_names, dynamic_axes = self._constitute_input_name_and_dynamic_axes_for_encoder_only() 
            runtime_model_child_dirpath = f"{self.runtime_model_path}/encoder"
            runtime_model_path = f'{runtime_model_child_dirpath}/model.onnx'
        else:
            input_names, dynamic_axes = self._constitute_input_name_and_dynamic_axes() 
            runtime_model_child_dirpath = self.runtime_model_path
            runtime_model_path = f'{runtime_model_child_dirpath}/model.onnx'
        pathlib.Path(runtime_model_child_dirpath).mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            model=model,
            args=tuple([v for v in encoded.values()]),
            f=runtime_model_path,
            export_params=True,
            do_constant_folding=False,
            opset_version=14,
            input_names=input_names,
            output_names=['output_logits'],
            dynamic_axes=dynamic_axes,
            verbose=True
        )

        onnx_model = onnx.load(runtime_model_path)

        out_graph = onnx.helper.printable_graph(onnx_model.graph)
        print(f"#### onnx.helper.printable_graph ( {runtime_model_path} ) ####")
        print(out_graph[:200])
        print("...")
        print(out_graph[-200:])
        print(f"#### END-OF-onnx.helper.printable_graph() ####")
        
        return 0


class TorchHfModelConverter(TorchModelConverter):
    def __init__(self, model_path, runtime_model_path, max_length, tokenizer_path_or_pt_model_name, lang, device=None, encoder_only=False):
        super(TorchHfModelConverter, self).__init__(model_path, runtime_model_path, max_length, lang, encoder_only)
        self.tokenizer_path_or_pt_model_name = tokenizer_path_or_pt_model_name
        self.device = device

        # TODO n_gpu_req option
        if not device:
            self.device, _ = prepare_device(n_gpu_req=-1)
        else:
            self.device = device

    def _make_hf_encoding_input(self, config):
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.tokenizer_path_or_pt_model_name,
            config=config,
            # add_prefix_space=False # roberta
        )
        encoded = tokenizer.batch_encode_plus(
            TorchModelConverter.LANG_EXAMPLES[self.lang],
            max_length=self.max_length,
            padding=DEFAULT_PADDING,
            truncation=True,
            pad_to_max_length=True,
            return_tensors="pt"
        ).to(self.device)

        return encoded


class TorchPretrainedModelConverter(TorchHfModelConverter):
    def __init__(self, model_path, runtime_model_path, max_length, tokenizer_path_or_pt_model_name, lang, device, encoder_only):
        super(TorchPretrainedModelConverter, self).__init__(model_path, runtime_model_path, max_length, tokenizer_path_or_pt_model_name, lang, device, encoder_only)
    
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


class TorchModelForSeq2SeqLmModelConverter(TorchHfModelConverter):
    def __init__(self, model_path, runtime_model_path, max_length, tokenizer_path_or_pt_model_name, lang, device, encoder_only=False):
        super(TorchModelForSeq2SeqLmModelConverter, self).__init__(model_path, runtime_model_path, max_length, tokenizer_path_or_pt_model_name, lang, device, encoder_only)

    def _constitute_input_name_and_dynamic_axes(self):
        input_names = ['input_ids', 'attention_mask']
        dynamic_axes = {
            'input_ids': {0:'batch_size', 1:'sequence'},
            'attention_mask': {0:'batch_size', 1:'sequence'},
            'output_logits': {0:'batch_size', 1: 'token_size'}
        }
        return input_names, dynamic_axes

    def __call__(self):
        config_path = f"{self.model_path}/config.json"
        # config = AutoConfig.from_pretrained(self.model_path, torchscript=False)
        config = AutoConfig.from_pretrained(config_path, torchscript=False)

        model = AutoModelForSeq2SeqLM.from_pretrained(
            str(self.model_path),
            config=config
        ).to(self.device)
        if self.encoder_only is True:
            model = model.get_encoder()

        model.eval()
        model.to(self.device)

        with torch.no_grad():
            encoded = self._make_hf_encoding_input(config)
            ret_code = self._export(model, encoded)

            return ret_code
