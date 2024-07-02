import datasets
import numpy as np
from tweak.metrics.perplexity import perplexity

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class SequenceScores(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description="",
            citation="",
            inputs_description="",
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Sequence(datasets.Value("float32"))),
                    # "predictions": datasets.Sequence(datasets.Value("int64")),
                    "references": datasets.Sequence(datasets.Value("int64"))
                }
            )
        )

    # def _compute(self, predictions, references):
    #     metric = perplexity(predictions, references, ignore_index=-100)
    #     return {"perplexity": metric}

    def _compute(self, label_list, predictions, references):

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, references)
        ]
        true_references = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, references)
        ]

        true_references = np.array(true_references).flatten()
        true_predictions = np.array(true_predictions).flatten()

        return {
            "accuracy": accuracy_score(true_references, true_predictions),
            "precision": precision_score(true_references, true_predictions, average='macro'),
            "recall": recall_score(true_references, true_predictions, average='macro'),
            "f1": f1_score(true_references, true_predictions, average='macro'),
        }
