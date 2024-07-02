import datasets

from dataclasses import dataclass

from tunip.env import NAUTS_HOME, TAGGED_JSONL_SUFFIX
from tunip.logger import init_logging_handler_for_klass
from tunip.service_config import get_service_config
from tunip.snapshot_utils import SnapshotPathProvider
from tunip.spark_utils import SparkConnector


_CITATION = """\
"""

_DESCRIPTION = """\
"""


@dataclass
class DatasetItemRecoTargetFeedbackConfig(datasets.BuilderConfig):

    dataset_path: str = f"{NAUTS_HOME}/data"
    dataset_file_suffix = TAGGED_JSONL_SUFFIX


class DatasetItemRecoTargetFeedback(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIG_CLASS = DatasetItemRecoTargetFeedbackConfig

    BUILDER_CONFIGS = [
        DatasetItemRecoTargetFeedbackConfig(
            name="Dataset for Item Recommendation by Vector Search",
            version=datasets.Version("1.0.0"),
            description="Dataset for Item Vector Search including adhoc and static informations",
        )
    ]

    def __init__(self, **kwargs):
        super(DatasetItemRecoTargetFeedback, self).__init__(**kwargs)

        self.logger = init_logging_handler_for_klass(klass=self.__class__)
        self.service_config = get_service_config()

        task_name = 'generation'
        # domain_name = 'item_reco_txt'
        domain_name = 'query2item_intro'

        snapshot_path_provider = SnapshotPathProvider(self.service_config)

        self.corpus_path_train_path = snapshot_path_provider.latest(
            f"/user/{self.service_config.username}/mart/corpus.vec/{task_name}/{domain_name}"
        ) + "/" + "train"
        self.corpus_path_validation_path = snapshot_path_provider.latest(
            f"/user/{self.service_config.username}/mart/corpus.vec/{task_name}/{domain_name}"
        ) + "/" + "validation"
        self.corpus_path_test_path = snapshot_path_provider.latest(
            f"/user/{self.service_config.username}/mart/corpus.vec/{task_name}/{domain_name}"
        ) + "/" + "test"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "recommendation_sid": datasets.Value("string"),
                    "item_id": datasets.Value("string"),
                    "inputs_embeds": datasets.Sequence(datasets.Value("float32")),
                    "decoder_inputs_embeds": datasets.Sequence(datasets.Value("float32")),
                    "labels": datasets.Value("float32"),
                }
            ),
            supervised_keys=None,
            homepage="http://localhost",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        """Returns SplitGenerators."""

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"corpus_path": self.corpus_path_train_path},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"corpus_path": self.corpus_path_validation_path},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"corpus_path": self.corpus_path_test_path},
            ),
        ]

    def _generate_examples(self, corpus_path):
        self.logger.info("⏳ Generating examples from = %s", corpus_path)

        spark = SparkConnector.getOrCreate(local=True, spark_config={"spark.driver.memory": "4g", "spark.executor.cores": 4}) 
        corpus_df = spark.read.json(corpus_path)

        for guid, row in enumerate(corpus_df.collect()):
            obj = row.asDict()
            recommendation_sid = obj['sid']
            item_id = obj['item_id']

            adhoc_vector = obj['text_vector']
            static_vector = obj['next_vector']

            feedback = 1/3 * (obj['interaction'] + obj['passion'] + obj['promise'])

            yield guid, {
                "recommendation_sid": recommendation_sid,
                "item_id": item_id,
                "inputs_embeds": adhoc_vector,
                "decoder_inputs_embeds": static_vector,
                "labels": feedback,
            }
