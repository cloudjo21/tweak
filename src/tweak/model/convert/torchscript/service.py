import urllib.parse

from tunip.path.mart import MartTokenizerPath

from tweak.model.convert.requests import Torch2TorchScriptRequest
from tweak.model.convert.torchscript import TorchScriptPretrainedModelConverter #, TorchModelForTokenClassificationModelConverter
from tweak.model.convert.torchscript.runtime_path import RuntimePathProviderFactory


class TorchScriptModelConvertService:
    def __init__(self, service_config):
        self.service_config = service_config
    
    def __call__(self, conv_req: Torch2TorchScriptRequest):
        runtime_path_provider = RuntimePathProviderFactory.create(conv_req.model_type, user_name=self.service_config.username)
        model_path, runtime_model_path = runtime_path_provider.provide(conv_req)

        model_path = f"{self.service_config.local_prefix}{model_path}"
        runtime_model_path = f"{self.service_config.local_prefix}{runtime_model_path}"

        if conv_req.tokenizer_name:
            tokenizer_name = urllib.parse.quote(conv_req.tokenizer_name, safe='')
            tokenizer_path_or_pt_model_name = f"{self.service_config.local_prefix}/{MartTokenizerPath(user_name=self.service_config.username, tokenizer_name=tokenizer_name)}"
        else:
            tokenizer_path_or_pt_model_name = DEFAULT_PT_MODEL_NAME[conv_req.lang]

        if conv_req.model_type == 'hf.plm_model':
            model_converter = TorchScriptPretrainedModelConverter(model_path, runtime_model_path, conv_req.max_length, tokenizer_path_or_pt_model_name, conv_req.lang)
        
        # TODO
        # elif conv_req.model_type == 'hf.token_classification_model':
        #     model_converter = TorchScriptModelForTokenClassificationModelConverter(model_path, runtime_model_path, conv_req.max_length, tokenizer_path_or_pt_model_name, conv_req.lang)
        return model_converter()

