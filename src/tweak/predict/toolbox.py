import pickle
from dataclasses import dataclass

from tweak.predict.builds import (
    PredictionBuildForTokenTypeWord,
    PredictionBuildForLastHiddenState,
    PredictionBuildForTorchScriptLastHiddenState,
    PredictionBuildForTorchScriptLastHiddenStateWithZero,
    PredictionBuildForLastHiddenStateWithAttentionMask,
)
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

        # TODO resource materialization
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
        if predictor_config.predict_model_type in ['auto']:
            prediction_build_cls = PredictionBuildForLastHiddenState
        elif predictor_config.predict_model_type in ['torchscript'] and predictor_config.zero_padding:
            prediction_build_cls = PredictionBuildForTorchScriptLastHiddenStateWithZero
        elif predictor_config.predict_model_type in ['torchscript'] and not predictor_config.zero_padding:
            prediction_build_cls = PredictionBuildForTorchScriptLastHiddenState

        if predictor_config.predict_output_type == 'last_hidden_with_attention_mask':
            prediction_build_cls = PredictionBuildForLastHiddenStateWithAttentionMask

        return PredictionToolboxForPreTrainedModel(model, tokenizer, prediction_build_cls)
    