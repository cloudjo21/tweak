import torch

from typing import List

from transformers.modeling_outputs import (
    TokenClassifierOutput,
)
from transformers.tokenization_utils_base import BatchEncoding

from tunip.corpus_utils import CorpusToken

from tweak.predict.models import PreTrainedModelConfig
from tweak.predict.predictor import Predictor, PredictorConfig
from tweak.predict.predict_pretrained import PreTrainedModelPredictor
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

        # for PLM predictor
        if isinstance(predictor_config.model_config, PreTrainedModelConfig):
            return PreTrainedModelPredictor(predictor_config)

        # for down-stream task predictor
        task_type = predictor_config.model_config.task_type
        assert predictor_config.model_config.task_type in [t.name for t in TaskType]

        if TaskType[task_type] is TaskType.TOKEN_CLASSIFICATION:
            return TokenClassificationPredictor(predictor_config)
        raise UnsupportedPredictorException(f'unsupported task type for predictor: {task_type}')
        