import datasets

from datasets import load_metric
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from seqeval.scheme import IOB2 as IOB_FORMAT


class IobSequenceScores(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description="",
            citation="",
            inputs_description="",
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("int64")),
                    "references": datasets.Sequence(datasets.Value("int64"))
                }
            )
        )

    def _compute(self, label_list, predictions, references):
        # def _compute(self, *, predictions=None, references=None, **kwargs) -> Dict[str, Any]:

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, references)
        ]
        true_references = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, references)
        ]

        return {
            "accuracy": accuracy_score(true_references, true_predictions),
            "precision": precision_score(true_references, true_predictions, mode='strict', scheme=IOB_FORMAT),
            "recall": recall_score(true_references, true_predictions, mode='strict', scheme=IOB_FORMAT),
            "f1": f1_score(true_references, true_predictions, mode='strict', scheme=IOB_FORMAT),
        }
