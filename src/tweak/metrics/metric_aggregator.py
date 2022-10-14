import numpy as np

from typing import Dict


class WeightedMetricAggregator:
    def __init__(self, weights_dict: Dict[str, float]):
        self.weights_dict = weights_dict
        self.total_weights = sum([x for x in weights_dict.values()])

    def aggregate(self, major_metrics_dict: Dict[str, float]):
        return (
            np.sum(
                [
                    x * self.weights_dict[task_name]
                    for task_name, x in major_metrics_dict.items()
                ]
            )
            / self.total_weights
        )
