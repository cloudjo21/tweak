from tunip.service_config import get_service_config

from tweak.predict.models import ModelConfig, PreTrainedModelConfig
from tweak.predict.models.hf_auto import HFAutoModelForPreTrained, HFAutoModelForTokenClassification
from tweak.predict.models.torchscript import TorchScriptModelForPreTrained
from tweak.predict.models.triton import TritonClientModelForTokenClassification
from tweak.predict.resource_materialize import ResourceMaterializer
from tweak.task.task_set import TaskType


class UnsupportedTaskTypeForModels(Exception):
    pass


class ModelsForPreTrainedModelFactory:

    @classmethod
    def create(cls, predict_model_type: str, model_config: PreTrainedModelConfig):
        ResourceMaterializer.apply_for_hf_model(model_config, get_service_config())

        if predict_model_type == 'torchscript':
            return TorchScriptModelForPreTrained(model_config)
        else:
            return HFAutoModelForPreTrained(model_config)


class ModelsForTokenClassificationFactory:

    @classmethod
    def create(cls, predict_model_type: str, model_config: ModelConfig):
        if predict_model_type == 'triton':
            return TritonClientModelForTokenClassification(model_config) 
        else:
            ResourceMaterializer.apply_for_hf_model(model_config, get_service_config())
            return HFAutoModelForTokenClassification(model_config)


class ModelsFactory:

    @classmethod
    def create(cls, predict_model_type: str, model_config: ModelConfig):
        if isinstance(model_config, PreTrainedModelConfig):
            return ModelsForPreTrainedModelFactory.create(predict_model_type, model_config)

        if TaskType[model_config.task_type] == TaskType.TOKEN_CLASSIFICATION:
            return ModelsForTokenClassificationFactory.create(predict_model_type, model_config)
        # TODO
        # TaskType[model_config.task_type] == TaskType.SEQUENCE_CLASSIFICATION
        raise UnsupportedTaskTypeForModels(f'unsupported task type: {model_config.task_type}')
