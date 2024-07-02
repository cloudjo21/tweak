import datasets

# from datasets import load_metric


class SpanScores(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description="",
            citation="",
            inputs_description="",
            features=datasets.Features(
                {
                    ""
                }
            )
        )