from typing import Dict


from tweak.metrics import (
    REGRESSION_METRIC,
    SEQUENCE_CLASSIFICATION_METRIC,
    TOKEN_CLASSIFICATION_METRIC,
    CAUSAL_LM_METRIC,
)
from tweak.metrics.metric_computer import (
    MetricComputer,
    MetricComputerFactory
)
from tweak.task.task_set import (
    TaskSet,
    TaskType
)


class MetricBuilder:

    token_classification_metric = TOKEN_CLASSIFICATION_METRIC
    sequence_classification_metric = SEQUENCE_CLASSIFICATION_METRIC
    causal_lm_metric = CAUSAL_LM_METRIC
    regression_metric = REGRESSION_METRIC

    predifined_metrics = {
        TaskType.TOKEN_CLASSIFICATION: token_classification_metric,
        TaskType.SEQUENCE_CLASSIFICATION: sequence_classification_metric,
        TaskType.CAUSAL_LM: causal_lm_metric,
        TaskType.SEQ2SEQ_LM: causal_lm_metric,
        TaskType.REGRESSION: regression_metric
    }

    @classmethod
    def create(
        cls,
        taskset: TaskSet,
        label_list_dict: Dict[str, list]
    ) -> Dict[str, MetricComputer]:
        task2metric = dict()
        for task in iter(taskset):
            if task.task_type not in cls.predifined_metrics:
                raise ValueError(f"Not supported task_type: {task.task_type} for metrics")
            else:
                metric = MetricComputerFactory.create(
                    task_type=task.task_type,
                    label_list=label_list_dict[task.task_name]
                )
                task2metric[task.task_name] = metric

        return task2metric
