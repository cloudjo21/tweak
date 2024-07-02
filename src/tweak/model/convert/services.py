import urllib.parse

from tunip.path.mart import MartTokenizerPath, MartPretrainedModelPath

from tweak import LOGGER
from tweak.model.convert.requests import Torch2OnnxRequest
from tweak.model.convert import (
    DEFAULT_PT_MODEL_NAME,
    TorchPretrainedModelConverter,
    TorchModelForSeq2SeqLmModelConverter,
    TorchModelForTokenClassificationModelConverter
)
from tweak.model.convert.runtime_path import RuntimePathProviderFactory


class TorchModelConvertService:
    def __init__(self, service_config):
        self.service_config = service_config
    
    def __call__(self, conv_req: Torch2OnnxRequest):
        runtime_path_provider = RuntimePathProviderFactory.create(conv_req.model_type, user_name=self.service_config.username)
        model_path, runtime_model_path = runtime_path_provider.provide(conv_req)

        tokenizer_name = urllib.parse.quote(conv_req.tokenizer_name, safe='')

        model_path = f"{self.service_config.local_prefix}{model_path}"
        runtime_model_path = f"{self.service_config.local_prefix}{runtime_model_path}"

        if conv_req.tokenizer_name:
            tokenizer_name = urllib.parse.quote(conv_req.tokenizer_name, safe='')
            pt_model_name = urllib.parse.quote(conv_req.pt_model_name, safe='')
            tokenizer_path_or_pt_model_name = f"{self.service_config.local_prefix}/{MartPretrainedModelPath(user_name=self.service_config.username, model_name=pt_model_name)}/vocab"
            # tokenizer_name = urllib.parse.quote(conv_req.tokenizer_name, safe='')
            # tokenizer_path_or_pt_model_name = f"{self.service_config.local_prefix}/{MartTokenizerPath(user_name=self.service_config.username, tokenizer_name=tokenizer_name)}"
        else:
            tokenizer_path_or_pt_model_name = DEFAULT_PT_MODEL_NAME[conv_req.lang]

        if conv_req.model_type == 'hf.plm_model':
            model_converter = TorchPretrainedModelConverter(model_path, runtime_model_path, conv_req.max_length, tokenizer_path_or_pt_model_name, conv_req.lang, conv_req.device, conv_req.encoder_only)
        elif conv_req.model_type == 'hf.token_classification_model':
            model_converter = TorchModelForTokenClassificationModelConverter(model_path, runtime_model_path, conv_req.max_length, tokenizer_path_or_pt_model_name, conv_req.lang, conv_req.device)
        elif conv_req.model_type == 'hf.seq2seq_lm_model':
            model_converter = TorchModelForSeq2SeqLmModelConverter(model_path, runtime_model_path, conv_req.max_length, tokenizer_path_or_pt_model_name, conv_req.lang, conv_req.device, conv_req.encoder_only)
        return model_converter()
