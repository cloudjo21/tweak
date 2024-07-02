from tunip.service_config import get_service_config

from tweak.predict.models import ModelConfig, PreTrainedModelConfig
from tweak.predict.models.hf_auto import (
    HFAutoModelForPreTrained,
    HFAutoModelForSeq2SeqLM,
    HFAutoModelForTokenClassification,
)
from tweak.predict.models.torchscript import (
    TorchScriptModelForPreTrained,
    TorchScriptEncoderForPreTrained,
    TorchScriptEncoderForSeq2SeqLM,
    TorchScriptModelForSeq2SeqLM,
    TorchScriptModelForTokenClassification,
)
from tweak.predict.models.triton import (
    TritonClientModelForPreTrainedModelFactory,
    TritonClientModelForSeq2SeqLMFactory,
    TritonClientModelForTokenClassificationFactory,
)
from tweak.predict.resource_materialize import ResourceMaterializer
from tweak.task.task_set import TaskType


class UnsupportedPreTrainedEncoder(Exception):
    pass

class UnsupportedTaskTypeForModels(Exception):
    pass


class ModelsForPreTrainedModelFactory:

    @classmethod
    def create(cls, predict_model_type: str, model_config: PreTrainedModelConfig):
        ResourceMaterializer.apply_for_hf_model(model_config, get_service_config())

        if predict_model_type == 'triton':
            return TritonClientModelForPreTrainedModelFactory.create(model_config)
        elif predict_model_type == 'torchscript':
            return TorchScriptModelForPreTrained(model_config)
        else:
            return HFAutoModelForPreTrained(model_config)


class ModelsForPreTrainedEncoderFactory:
    @classmethod
    def create(cls, predict_model_type: str, model_config: PreTrainedModelConfig):
        ResourceMaterializer.apply_for_hf_model(model_config, get_service_config())

        if predict_model_type == 'torchscript':
            return TorchScriptEncoderForPreTrained(model_config)
        elif predict_model_type == 'triton':
            return TritonClientModelForPreTrainedModelFactory.create(model_config)
        else:
            raise UnsupportedPreTrainedEncoder(f"unsupported encoder: {model_config.model_name}, encoder_only={model_config.encoder_only}")
        # else:
        #     return HFAutoModelForPreTrained(model_config)


class ModelsForTokenClassificationFactory:

    @classmethod
    def create(cls, predict_model_type: str, model_config: ModelConfig):
        ResourceMaterializer.apply_for_hf_model(model_config, get_service_config())
        
        if predict_model_type == 'triton':
            return TritonClientModelForTokenClassificationFactory.create(model_config)
        elif predict_model_type == 'torchscript':
            return TorchScriptModelForTokenClassification(model_config) 
        else:
            return HFAutoModelForTokenClassification(model_config)


class ModelsForSeq2SeqLMFactory:

    @classmethod
    def create(cls, predict_model_type: str, model_config: ModelConfig):
        ResourceMaterializer.apply_for_hf_model(model_config, get_service_config())
        
        if predict_model_type == 'triton':
            return TritonClientModelForSeq2SeqLMFactory.create(model_config)
        elif predict_model_type == 'torchscript':
            if model_config.encoder_only is True:
                return TorchScriptEncoderForSeq2SeqLM(model_config)
            return TorchScriptModelForSeq2SeqLM(model_config) 
        else:
            return HFAutoModelForSeq2SeqLM(model_config)


class ModelsFactory:

    @classmethod
    def create(cls, predict_model_type: str, model_config: ModelConfig):
        if isinstance(model_config, PreTrainedModelConfig):
            if model_config.encoder_only is True:
                return ModelsForPreTrainedEncoderFactory.create(predict_model_type, model_config)
            else:
                return ModelsForPreTrainedModelFactory.create(predict_model_type, model_config)

        if TaskType[model_config.task_type] == TaskType.TOKEN_CLASSIFICATION:
            return ModelsForTokenClassificationFactory.create(predict_model_type, model_config)

        elif TaskType[model_config.task_type] == TaskType.SEQ2SEQ_LM:
            return ModelsForSeq2SeqLMFactory.create(predict_model_type, model_config)
        # TODO
        # TaskType[model_config.task_type] == TaskType.SEQUENCE_CLASSIFICATION
        else:
            raise UnsupportedTaskTypeForModels(f'unsupported task type: {model_config.task_type}')
