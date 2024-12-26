import pickle
from dataclasses import dataclass
from pydantic import BaseModel

from tunip.orjson_utils import *
from tweak.predict.builds import (
    PredictionBuildForLastHiddenState,
    PredictionBuildForHiddenStates,
    PredictionBuildForTorchScriptLastHiddenState,
    PredictionBuildForTorchScriptLastHiddenStateWithZero,
    PredictionBuildForTorchScriptLastHiddenStateForMeanPooling,
    PredictionBuildForTorchScriptLastHiddenStateForGlobalMeanPool,
    PredictionBuildForHuggingfaceTokenGen,
    PredictionBuildForTorchScriptTokenGen,
    PredictionBuildForLastHiddenStateForClsToken,
    PredictionBuildForLastHiddenStateWithAttentionMask,
    PredictionBuildForLastHiddenStateWithAttentionMaskForMeanPooling,
    PredictionBuildForTorchScriptTokenTypeWord,
    PredictionBuildForTritonLastHiddenState,
    PredictionBuildForTritonLastHiddenStateForClsToken,
    PredictionBuildForTritonLastHiddenStateForMeanPooling,
    PredictionBuildForTritonLastHiddenStateForGlobalMeanPooling,
    PredictionBuildForTritonTokenTypeWord,
    PredictionBuildForHuggingFaceTokenTypeWord
)
from tweak.predict.models.factory import ModelsFactory
from tweak.predict.predictor import PredictorConfig
from tweak.predict.tokenizers import TokenizersFactory


class PredictionToolbox(BaseModel):
    model: object
    tokenizer: object
    device: str
    on_remote: bool = False
    prediction_build_cls: object.__class__

    class Config:
        arbitrary_types_allowed = True

    @property
    def model_on_device(self):
        if not self.on_remote:
            self.model.model.to(self.device)
        return self.model


class PredictionToolBoxForTokenClassification(PredictionToolbox):
    label_list: list


class PredictionToolboxPackerForTokenClassification:

    @classmethod
    def pack(cls, predictor_config: PredictorConfig):

        tokenizer = TokenizersFactory.create(
            predictor_config.predict_tokenizer_type,
            predictor_config.tokenizer_config.json()
        )

        model = ModelsFactory.create(predictor_config.predict_model_type, predictor_config.predict_model_config)

        # TODO resource materialization
        label_list_path = f"{predictor_config.predict_model_config.model_path}/label_list.pickle"
        with open(label_list_path, "rb") as lf:
            label_list = pickle.load(lf)

        # TODO provide child class of PredictionBuild by factory
        if predictor_config.predict_model_type == "torchscript":
            prediction_build_cls = PredictionBuildForTorchScriptTokenTypeWord
        elif predictor_config.predict_model_type == "triton":
            prediction_build_cls = PredictionBuildForTritonTokenTypeWord
        else:
            prediction_build_cls = PredictionBuildForHuggingFaceTokenTypeWord

        return PredictionToolBoxForTokenClassification(
            model=model,
            tokenizer=tokenizer,
            device=predictor_config.device,
            on_remote=predictor_config.is_on_remote,
            prediction_build_cls=prediction_build_cls,
            label_list=label_list
        )


class PredictionToolboxForSeq2SeqLM(PredictionToolbox):
    label_list: list

class PredictionToolboxPackerForSeq2SeqLM:

    @classmethod
    def pack(cls, predictor_config: PredictorConfig):

        tokenizer = TokenizersFactory.create(
            predictor_config.predict_tokenizer_type,
            predictor_config.tokenizer_config.json()
        )

        model = ModelsFactory.create(predictor_config.predict_model_type, predictor_config.predict_model_config)

        # TODO resource materialization
        label_list_path = f"{predictor_config.predict_model_config.model_path}/label_list.pickle"
        with open(label_list_path, "rb") as lf:
            label_list = pickle.load(lf)

        assert label_list

        prediction_build_cls = None

        if predictor_config.predict_model_type in ["auto"] and predictor_config.predict_output_type == "last_hidden":
            prediction_build_cls = PredictionBuildForLastHiddenState
        elif predictor_config.predict_model_type in ["auto"] and predictor_config.predict_output_type == "hidden":
            prediction_build_cls = PredictionBuildForHiddenStates
        elif predictor_config.predict_model_type in ["torchscript"] and predictor_config.predict_output_type == "last_hidden":
            prediction_build_cls = PredictionBuildForLastHiddenState
        elif predictor_config.predict_model_type == "torchscript":
            prediction_build_cls = PredictionBuildForTorchScriptTokenGen
        # TODO support triton prediction
        # elif predictor_config.predict_model_type == "triton":
        #     prediction_build_cls = PredictionBuildForTritonTokenGeneration
        else:
            prediction_build_cls = PredictionBuildForHuggingfaceTokenGen

        return PredictionToolboxForSeq2SeqLM(
            model=model,
            tokenizer=tokenizer,
            device=predictor_config.device,
            on_remote=predictor_config.is_on_remote,
            prediction_build_cls=prediction_build_cls,
            label_list=label_list
        )


class PredictionToolboxForPreTrainedModel(PredictionToolbox):
    pass

class PredictionToolboxForPreTrainedEncoder(PredictionToolbox):
    pass

class PredictionToolboxForSeq2SeqLMEncoder(PredictionToolbox):
    pass

class PredictionToolboxPackerForPreTrainedModel:

    @classmethod
    def pack(cls, predictor_config: PredictorConfig):
        tokenizer = TokenizersFactory.create(
            predictor_config.predict_tokenizer_type,
            predictor_config.tokenizer_config.json()
        )

        model = ModelsFactory.create(
            predictor_config.predict_model_type, predictor_config.predict_model_config
        )

        # TODO provide child class of PredictionBuild by factory
        # TODO refactoring considering predict_model_type:94

        if predictor_config.predict_model_type in ['auto'] and \
                predictor_config.predict_output_type == 'last_hidden_with_attention_mask':
            prediction_build_cls = PredictionBuildForLastHiddenStateWithAttentionMask
        elif predictor_config.predict_model_type in ['auto'] and predictor_config.predict_output_type == 'last_hidden_with_attention_mask.mean_pooling':
            prediction_build_cls = PredictionBuildForLastHiddenStateWithAttentionMaskForMeanPooling
        elif predictor_config.predict_model_type in ['auto'] and predictor_config.predict_output_type == 'last_hidden.cls_token':
            prediction_build_cls = PredictionBuildForLastHiddenStateForClsToken
        elif predictor_config.predict_model_type in ['auto']:
            prediction_build_cls = PredictionBuildForLastHiddenState
        elif predictor_config.predict_model_type in ['torchscript'] and predictor_config.predict_output_type == 'last_hidden.global_mean_pooling':
            prediction_build_cls = PredictionBuildForTorchScriptLastHiddenStateForGlobalMeanPool
        elif predictor_config.predict_model_type in ['torchscript'] and predictor_config.predict_output_type == 'last_hidden.mean_pooling':
            prediction_build_cls = PredictionBuildForTorchScriptLastHiddenStateForMeanPooling
        elif predictor_config.predict_model_type in ['torchscript'] and predictor_config.zero_padding:
            prediction_build_cls = PredictionBuildForTorchScriptLastHiddenStateWithZero
        elif predictor_config.predict_model_type in ['torchscript'] and not predictor_config.zero_padding:
            prediction_build_cls = PredictionBuildForTorchScriptLastHiddenState
        elif predictor_config.predict_model_type in ['triton'] and predictor_config.predict_output_type == 'last_hidden.global_mean_pooling':
            prediction_build_cls = PredictionBuildForTritonLastHiddenStateForGlobalMeanPooling
        elif predictor_config.predict_model_type in ['triton'] and predictor_config.predict_output_type == 'last_hidden.mean_pooling':
            prediction_build_cls = PredictionBuildForTritonLastHiddenStateForMeanPooling
        elif predictor_config.predict_model_type in ['triton'] and predictor_config.predict_output_type == 'last_hidden.cls_token':
            prediction_build_cls = PredictionBuildForTritonLastHiddenStateForClsToken
        elif predictor_config.predict_model_type in ['triton']:
            prediction_build_cls = PredictionBuildForTritonLastHiddenState

        return PredictionToolboxForPreTrainedModel(
            model=model,
            tokenizer=tokenizer,
            device=predictor_config.device,
            on_remote=predictor_config.is_on_remote,
            prediction_build_cls=prediction_build_cls
        )
    

class PredictionToolboxPackerForPreTrainedEncoder:

    @classmethod
    def pack(cls, predictor_config: PredictorConfig):
        model = ModelsFactory.create(
            predictor_config.predict_model_type, predictor_config.predict_model_config
        )

        tokenizer = TokenizersFactory.create(
            predictor_config.predict_tokenizer_type,
            predictor_config.tokenizer_config.json()
        )

        if predictor_config.predict_model_type in ['torchscript'] and predictor_config.predict_output_type == 'last_hidden.global_mean_pooling':
            prediction_build_cls = PredictionBuildForTorchScriptLastHiddenStateForGlobalMeanPool
        elif predictor_config.predict_model_type in ['torchscript'] and predictor_config.predict_output_type == 'last_hidden.mean_pooling':
            prediction_build_cls = PredictionBuildForTorchScriptLastHiddenStateForMeanPooling
        elif predictor_config.predict_model_type in ['torchscript'] and predictor_config.zero_padding:
            prediction_build_cls = PredictionBuildForTorchScriptLastHiddenStateWithZero
        elif predictor_config.predict_model_type in ['torchscript'] and not predictor_config.zero_padding:
            prediction_build_cls = PredictionBuildForTorchScriptLastHiddenState
        elif predictor_config.predict_model_type in ['triton'] and predictor_config.predict_output_type == 'last_hidden.global_mean_pooling':
            prediction_build_cls = PredictionBuildForTritonLastHiddenStateForGlobalMeanPooling
        elif predictor_config.predict_model_type in ['triton'] and predictor_config.predict_output_type == 'last_hidden.mean_pooling':
            prediction_build_cls = PredictionBuildForTritonLastHiddenStateForMeanPooling
        elif predictor_config.predict_model_type in ['triton']:
            prediction_build_cls = PredictionBuildForTritonLastHiddenState
        else:
            prediction_build_cls = PredictionBuildForLastHiddenState

        return PredictionToolboxForPreTrainedEncoder(
            model=model,
            tokenizer=tokenizer,
            device=predictor_config.device,
            on_remote=predictor_config.is_on_remote,
            prediction_build_cls=prediction_build_cls
        )


class PredictionToolboxPackerForSeq2SeqLMEncoder(PredictionToolbox):

    @classmethod
    def pack(cls, predictor_config: PredictorConfig):
        tokenizer = TokenizersFactory.create(
            predictor_config.predict_tokenizer_type,
            predictor_config.tokenizer_config.json()
        )

        model = ModelsFactory.create(
            predictor_config.predict_model_type, predictor_config.predict_model_config
        )

        if predictor_config.predict_model_type in ['torchscript'] and predictor_config.predict_output_type == 'last_hidden.global_mean_pooling':
            prediction_build_cls = PredictionBuildForTorchScriptLastHiddenStateForGlobalMeanPool
        elif predictor_config.predict_model_type in ['torchscript'] and predictor_config.predict_output_type == 'last_hidden.mean_pooling':
            prediction_build_cls = PredictionBuildForTorchScriptLastHiddenStateForMeanPooling
        elif predictor_config.predict_model_type in ['torchscript'] and predictor_config.zero_padding:
            prediction_build_cls = PredictionBuildForTorchScriptLastHiddenStateWithZero
        elif predictor_config.predict_model_type in ['torchscript'] and not predictor_config.zero_padding:
            prediction_build_cls = PredictionBuildForTorchScriptLastHiddenState
        elif predictor_config.predict_model_type in ['triton'] and predictor_config.predict_output_type == 'last_hidden.global_mean_pooling':
            prediction_build_cls = PredictionBuildForTritonLastHiddenStateForGlobalMeanPooling
        elif predictor_config.predict_model_type in ['triton'] and predictor_config.predict_output_type == 'last_hidden.mean_pooling':
            prediction_build_cls = PredictionBuildForTritonLastHiddenStateForMeanPooling
        elif predictor_config.predict_model_type in ['triton']:
            prediction_build_cls = PredictionBuildForTritonLastHiddenState
        else:
            prediction_build_cls = PredictionBuildForLastHiddenState

        return PredictionToolboxForSeq2SeqLMEncoder(
            model=model,
            tokenizer=tokenizer,
            device=predictor_config.device,
            on_remote=predictor_config.is_on_remote,
            prediction_build_cls=prediction_build_cls
        )
