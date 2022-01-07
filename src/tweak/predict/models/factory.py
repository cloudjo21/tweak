from tweak.predict.models import ModelConfig
from tweak.predict.models.hf_auto import HFAutoModelForTokenClassification
from tweak.predict.models.triton import TritonClientModelForTokenClassification


class ModelsForTokenClassificationFactory:
    @classmethod
    def create(cls, predict_model_type:str, config: str):

        model_config = ModelConfig.parse_raw(config)

        if predict_model_type == 'triton':
            return TritonClientModelForTokenClassification(model_config)
        
        return HFAutoModelForTokenClassification(model_config)
