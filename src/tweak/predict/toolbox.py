import pickle

from dataclasses import dataclass

from tweak.predict.builds import PredictionBuild, PredictionBuildForTokenTypeWord
from tweak.predict.models.factory import ModelsForTokenClassificationFactory
from tweak.predict.predictor import PredictorConfig
from tweak.predict.tokenizers import TokenizersFactory


@dataclass
class PredictionToolbox:
    model: object
    label_list: list
    tokenizer: object
    prediction_build_cls: object.__class__


class PredictionToolboxPackerForTokenClassification:

    @classmethod
    def pack(self, predict_config: PredictorConfig):

        tokenizer = TokenizersFactory.create(
            predict_config.predict_tokenizer_type,
            predict_config.tokenizer_config.json()
        )

        model = ModelsForTokenClassificationFactory.create(
            predict_config.predict_model_type,
            predict_config.model_config.json()
        )

        label_list_path = f"{predict_config.model_config.model_path}/label_list.pickle"
        with open(label_list_path, "rb") as lf:
            label_list = pickle.load(lf)
        
        # TODO provide child class of PredictionBuild by factory
        prediction_build_cls = PredictionBuildForTokenTypeWord

        return PredictionToolbox(
            model, label_list, tokenizer, prediction_build_cls
        )
