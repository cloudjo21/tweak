from tunip.service_config import get_service_config

from tweak.predict.models import ModelConfig
from tweak.predict.models.hf_auto import HFAutoModelForTokenClassification
from tweak.predict.models.triton import TritonClientModelForTokenClassification
from tweak.task.task_set import TaskType


class UnsupportedTaskTypeForModels(Exception):
    pass


class ModelsForTokenClassificationFactory:
    @classmethod
    def create(cls, predict_model_type: str, model_config: ModelConfig):

        service_config = get_service_config()
        model_config.model_path = f"{service_config.filesystem_prefix}/{model_config.model_path}"

        if predict_model_type == 'triton':
            return TritonClientModelForTokenClassification(model_config)
        
        return HFAutoModelForTokenClassification(model_config)


class ModelsFactory:
    @classmethod
    def create(cls, predict_model_type: str, model_config: ModelConfig):
        if TaskType[model_config.task_type] == TaskType.TOKEN_CLASSIFICATION:
            return ModelsForTokenClassificationFactory.create(predict_model_type, model_config)
        # TODO
        # TaskType[model_config.task_type] == TaskType.SEQUENCE_CLASSIFICATION
        raise UnsupportedTaskTypeForModels(f'unsupported task type: {model_config.task_type}')
