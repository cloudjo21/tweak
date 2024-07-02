import os
import urllib.parse

from copy import deepcopy
from typing import Optional

from transformers.modeling_outputs import (
    TokenClassifierOutput,
)

from tunip.path_utils import TaskPath
from tunip.service_config import get_service_config

from tweak.predict.models import PreTrainedModelConfig
from tweak.predict.predictor import Predictor, PredictorConfig
from tweak.predict.predict_pretrained import PreTrainedModelPredictor
from tweak.predict.predict_pretrained_encoder import PreTrainedEncoderPredictor
from tweak.predict.predict_seq2seq_lm import Seq2SeqLMPredictor
from tweak.predict.predict_seq2seq_lm_encoder import Seq2SeqLMEncoderPredictor
from tweak.predict.predict_token_classification import TokenClassificationPredictor
from tweak.task.task_set import TaskType


class PredictorForTokenClassification(Predictor):

    def __init__(self, pred_box):
        pass

    def predict(self):
        pass


class UnsupportedPredictorException(Exception):
    pass


class PredictorFactory:

    @classmethod
    def create(cls, predictor_config: PredictorConfig):

        cloned_config = deepcopy(predictor_config)

        # for PLM predictor
        if isinstance(cloned_config.predict_model_config, PreTrainedModelConfig):
            if cloned_config.predict_model_config.encoder_only is True:
                return PreTrainedEncoderPredictor(cloned_config)
            else:
                return PreTrainedModelPredictor(cloned_config)

        # for down-stream task predictor
        task_type = cloned_config.predict_model_config.task_type
        assert cloned_config.predict_model_config.task_type in [t.name for t in TaskType]

        if TaskType[task_type] is TaskType.SEQ2SEQ_LM:
            if cloned_config.predict_model_config.encoder_only is True:
                return Seq2SeqLMEncoderPredictor(cloned_config)
            return Seq2SeqLMPredictor(cloned_config)
        elif TaskType[task_type] is TaskType.TOKEN_CLASSIFICATION:
            return TokenClassificationPredictor(cloned_config)
        else:
            raise UnsupportedPredictorException(f'unsupported task type for predictor: {task_type}')
        
    
class SimplePredictorFactory:

    @classmethod
    def create(
        cls,
        model_name:str,
        plm:bool=True,
        username:Optional[str]=None,
        predict_output_type:str="last_hidden",
        tokenizer_type:Optional[str]=None,
        model_type:str="torchscript",
        device:str="cpu",
        encoder_only:bool=False,
        max_length:int=128,
        zero_padding:bool=False,
        domain_name:Optional[str]=None,
        task_name:Optional[str]=None,
        snapshot_dt:Optional[str]=None
    ):

        if not username:
            username = get_service_config().username

        if not tokenizer_type:
            tokenizer_type = "auto"

        if plm is False:
            assert domain_name is not None
            assert task_name is not None
            assert snapshot_dt is not None
            model_root_path = str(TaskPath(username, domain_name, snapshot_dt, task_name))
        else:
            model_root_path = f"/user/{username}/mart/plm/models/{urllib.parse.quote(model_name, safe='')}"

        if model_type == "torchscript":
            model_path_typed = model_root_path + os.sep + "torchscript"
        else:
            model_path_typed = model_root_path

        if encoder_only is True:
            model_path_encoded_or_not = model_path_typed + os.sep + "encoder"
        else:
            model_path_encoded_or_not = model_path_typed

        tokenizer_path = f"{model_root_path}/vocab"

        predict_config = {
            "predict_tokenizer_type": tokenizer_type,
            "predict_model_type": model_type,
            "predict_output_type": predict_output_type,
            "device": device,
            "zero_padding": zero_padding,
            "predict_model_config": {
                "model_path": model_path_encoded_or_not,
                "model_name": model_name,
                "encoder_only": encoder_only
            },
            "tokenizer_config": {
                "model_path": model_root_path,
                "path": tokenizer_path,
                "max_length": max_length
            }
        }

        pred_config_obj = PredictorConfig.model_validate(predict_config)
        predictor = PredictorFactory.create(pred_config_obj)

        return predictor
