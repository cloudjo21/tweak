import tweak.predict.models.triton.backend as triton_backend

from tweak.predict.models import ModelConfig, PreTrainedModelConfig


class TritonClientModelForTokenClassificationFactory:

    @classmethod
    def create(cls, model_config: ModelConfig):
        if model_config.remote_backend == 'torchscript':
            return triton_backend.torchscript.TritonClientModelForTokenClassification(model_config)
        else:
            return triton_backend.onnx.TritonClientModelForTokenClassification(model_config)


class TritonClientModelForSeq2SeqLMFactory:

    @classmethod
    def create(cls, model_config: ModelConfig):
        if model_config.remote_backend == 'torchscript':
            return triton_backend.torchscript.TritonClientModelForSeq2SeqLM(model_config)
        else:
            return triton_backend.onnx.TritonClientModelForSeq2SeqLM(model_config)


class TritonClientModelForPreTrainedModelFactory:

    @classmethod
    def create(cls, model_config: PreTrainedModelConfig):
        if model_config.remote_backend == 'torchscript':
            if model_config.encoder_only is True:
                return triton_backend.torchscript.TritonClientModelForPreTrainedModelEncoder(model_config)
            else:
                return triton_backend.torchscript.TritonClientModelForPreTrainedModel(model_config)
        else:
            if model_config.encoder_only is True:
                return triton_backend.onnx.TritonClientModelForPreTrainedModelEncoder(model_config)
            else:
                return triton_backend.onnx.TritonClientModelForPreTrainedModel(model_config)
