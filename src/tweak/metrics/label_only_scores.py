import datasets

from datasets import load_metric
from seqeval.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support


class LabelOnlyScores(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description="",
            citation="",
            inputs_description="",
            features=datasets.Features(
                {
                    "predictions": datasets.Value("int64"),
                    "references": datasets.Value("int64")
                }
            )
        )

    def _compute(self, predictions, references):
        # def _compute(self, *, predictions=None, references=None, **kwargs) -> Dict[str, Any]:

        precision, recall, f1, _ = precision_recall_fscore_support(
            references, predictions, average='weighted'
        )
        return {
            "accuracy": accuracy_score(references, predictions),
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
