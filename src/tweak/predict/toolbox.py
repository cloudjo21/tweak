import pickle

from dataclasses import dataclass

from tweak.predict.builds import PredictionBuildForTokenTypeWord, PredictionBuildForLastHiddenState
from tweak.predict.models.factory import ModelsFactory
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
    def pack(cls, predictor_config: PredictorConfig):

        tokenizer = TokenizersFactory.create(
            predictor_config.predict_tokenizer_type,
            predictor_config.tokenizer_config.json()
        )

        model = ModelsFactory.create(predictor_config.predict_model_type, predictor_config.model_config)

        label_list_path = f"{predictor_config.model_config.model_path}/label_list.pickle"
        with open(label_list_path, "rb") as lf:
            label_list = pickle.load(lf)
        
        # TODO provide child class of PredictionBuild by factory
        prediction_build_cls = PredictionBuildForTokenTypeWord

        return PredictionToolbox(
            model, label_list, tokenizer, prediction_build_cls
        )


@dataclass
class PredictionToolboxForPreTrainedModel:
    model: object
    tokenizer: object
    prediction_build_cls: object.__class__


class PredictionToolboxPackerForPreTrainedModel:

    @classmethod
    def pack(cls, predictor_config: PredictorConfig):
        tokenizer = TokenizersFactory.create(
            predictor_config.predict_tokenizer_type,
            predictor_config.tokenizer_config.json()
        )

        model = ModelsFactory.create(
            predictor_config.predict_model_type, predictor_config.model_config
        )

        # TODO provide child class of PredictionBuild by factory
        prediction_build_cls = PredictionBuildForLastHiddenState

        return PredictionToolboxForPreTrainedModel(model, tokenizer, prediction_build_cls)
    