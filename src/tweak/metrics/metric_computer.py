import datasets
import numpy as np
import torch

from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional
from torch.utils.data.dataloader import DataLoader
from transformers.trainer_utils import EvalPrediction

from tweak.task.task_set import TaskType
from tweak.metrics import (
    REGRESSION_METRIC,
    SEQUENCE_CLASSIFICATION_METRIC,
    TOKEN_CLASSIFICATION_METRIC,
    CAUSAL_LM_METRIC
)
from tweak.metrics.perplexity import perplexity


@dataclass
class MetricComputeRequest:
    task_name: str
    eval_pred: EvalPrediction
    eval_dataloader: DataLoader = None


class MetricComputer:
    def __init__(self, metric: datasets.Metric, label_list: list):
        self.metric = metric
        self.label_list = label_list

    @abstractmethod
    def __call__(self, eval_pred: EvalPrediction, task_name: Optional[str]=None):
        # TODO raise exception if there is no task_name when use_mtl is True
        # TODO at first, consider MetricBuilder
        pass


class TokenClassificationMetricComputer(MetricComputer):
    def __init__(
        self,
        label_list: list,
        metric: datasets.Metric = TOKEN_CLASSIFICATION_METRIC
    ):
        super(TokenClassificationMetricComputer, self).__init__(
            metric,
            label_list
        )

    def __call__(self, eval_pred: EvalPrediction, task_name: Optional[str]=None):
        logits = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
        labels = eval_pred.label_ids
        # logits, labels, _ = eval_pred
        predictions = np.argmax(logits, axis=2)

        # iob sequence metric
        return self.metric.compute(
            label_list=self.label_list,
            predictions=predictions,
            references=labels
        )


class SequenceClassificationMetricComputer(MetricComputer):
    def __init__(
        self,
        label_list: list,
        metric: datasets.Metric = SEQUENCE_CLASSIFICATION_METRIC
    ):
        super(SequenceClassificationMetricComputer, self).__init__(
            metric,
            label_list
        )

    def __call__(self, eval_pred: EvalPrediction, task_name: Optional[str]=None):
        logits = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
        labels = eval_pred.label_ids
        # logits, labels, _ = eval_pred
        predictions = np.argmax(logits, axis=-1)

        # label only metric
        return self.metric.compute(
            predictions=predictions,
            references=labels
        )


class CausalLMMetricComputer(MetricComputer):
    def __init__(
        self,
        label_list: list,
        metric: datasets.Metric = CAUSAL_LM_METRIC
    ):
        super(CausalLMMetricComputer, self).__init__(
            metric,
            label_list
        )

    def __call__(self, eval_pred: EvalPrediction, task_name: Optional[str]=None):
        # get logits if use_return_dict or not
        logits = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
        labels = eval_pred.label_ids

        metric = perplexity(torch.tensor(logits, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64), ignore_index=-100)
        return {"perplexity": metric}

        # return self.metric.compute(predictions=logits, references=labels)

        # predictions = np.argmax(logits, axis=2)
        # #  metric
        # return self.metric.compute(
        #     label_list=self.label_list,
        #     predictions=predictions,
        #     references=labels
        # )

# TODO
class SpanOffsetsMetricComputer(MetricComputer):
    def __init__(self, label_list: list, metric: datasets.Metric):
        pass


class InverseClozeTaskMetricComputer(MetricComputer):
    def __init__(
        self,
        label_list: list,
        metric: datasets.Metric=SEQUENCE_CLASSIFICATION_METRIC
    ):
        super(InverseClozeTaskMetricComputer, self).__init__(
            metric,
            label_list
        )
    
    def __call__(self, eval_pred: EvalPrediction, task_name: Optional[str]=None):
        logits = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
        labels = eval_pred.label_ids
        # logits, label_sets, _ = eval_pred
        predictions = np.argmax(logits, axis=-1)
        # max_score, max_indexes = torch.max(logits, dim=1)
        # correct_counts = (max_indexes == positive_labels).sum()

        # TODO
        # 0: positive labels, 1: hard negative labels
        positive_labels = labels

        # label only metric
        return self.metric.compute(
            predictions=predictions,
            references=positive_labels
        )

class RegressionTaskMetricComputer(MetricComputer):
    def __init__(
        self,
        label_list: list,
        metric: datasets.Metric = REGRESSION_METRIC
    ):
        super(RegressionTaskMetricComputer, self).__init__(
            metric,
            label_list
        )

    def __call__(self, eval_pred: EvalPrediction, task_name: Optional[str]=None):
        predictions = eval_pred.predictions[1] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
        predictions = np.array([p.tolist() for p in eval_pred.predictions]).reshape(-1)
        labels = eval_pred.label_ids

        return self.metric.compute(
            predictions=predictions,
            references=labels
        )


class MetricComputerFactory:

    metric_computers = {
        TaskType.TOKEN_CLASSIFICATION: TokenClassificationMetricComputer,
        TaskType.SEQUENCE_CLASSIFICATION: SequenceClassificationMetricComputer,
        TaskType.CAUSAL_LM: CausalLMMetricComputer,
        TaskType.SEQ2SEQ_LM: CausalLMMetricComputer,
        TaskType.INVERSE_CLOZE_TASK: InverseClozeTaskMetricComputer,
        TaskType.REGRESSION: RegressionTaskMetricComputer 
    }

    @classmethod
    def create(cls, task_type: TaskType, label_list: list) -> MetricComputer:
        try:
            klass = cls.metric_computers[task_type]
        except IndexError as ie:
            raise ValueError(
                f"Invalid task type: {task_type.name} "
                "while creating metric computer"
            ) from ie
        return klass(label_list)
