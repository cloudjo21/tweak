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
class DatasetItemRecoTxtConfig(datasets.BuilderConfig):

    dataset_path: str = f"{NAUTS_HOME}/data"
    dataset_file_suffix = TAGGED_JSONL_SUFFIX


class DatasetItemRecoTxt(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIG_CLASS = DatasetItemRecoTxtConfig

    BUILDER_CONFIGS = [
        DatasetItemRecoTxtConfig(
            name="Dataset for Item Recommendation by Vector Search",
            version=datasets.Version("1.0.0"),
            description="Dataset for Item Vector Search including adhoc and static informations",
        )
    ]

    def __init__(self, **kwargs):
        super(DatasetItemRecoTxt, self).__init__(**kwargs)

        self.logger = init_logging_handler_for_klass(klass=self.__class__)
        self.service_config = get_service_config()

        task_name = 'generation'
        domain_name = 'query2item_intro'

        snapshot_path_provider = SnapshotPathProvider(self.service_config)

        self.corpus_path_train_path = snapshot_path_provider.latest(
            f"/user/{self.service_config.username}/mart/corpus.vec/{task_name}/{domain_name}"
        )
        self.corpus_path_dev_path = snapshot_path_provider.latest(
            f"/user/{self.service_config.username}/mart/corpus.vec/{task_name}/{domain_name}"
        )
        self.corpus_path_test_path = snapshot_path_provider.latest(
            f"/user/{self.service_config.username}/mart/corpus.vec/{task_name}/{domain_name}"
        )

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "recommendation_sid": datasets.Value("string"),
                    "item_id": datasets.Value("string"),
                    "inputs_embeds": datasets.Sequence(datasets.Value("float32")),
                    "decoder_inputs_embeds": datasets.Sequence(datasets.Value("float32")),
                    "item_description": datasets.Value("string"),
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
                gen_kwargs={"corpus_path": self.corpus_path_dev_path},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"corpus_path": self.corpus_path_test_path},
            ),
        ]

    def _generate_examples(self, corpus_path):
        self.logger.info("⏳ Generating examples from = %s", corpus_path)

        spark = SparkConnector.getOrCreate(local=True, spark_config={"spark.driver.memory": "4g", "spark.executor.cores": 4}) 

        if 'dev' in corpus_path:
            # corpus_df = spark.read.json(corpus_path).sample(fraction=0.1, seed=21)
            corpus_df = spark.read.json(corpus_path).take(200)
        elif 'test' in corpus_path:
            # corpus_df = spark.read.json(corpus_path).sample(fraction=0.1, seed=42)
            corpus_df = spark.read.json(corpus_path).take(200)
        else:
            # corpus_df = spark.read.json(corpus_path)
            corpus_df = spark.read.json(corpus_path).take(2000)

        for guid, row in enumerate(corpus_df):
        # for guid, row in enumerate(corpus_df.collect()):
            obj = row.asDict()
            recommendation_sid = obj['sid']
            item_id = obj['item_id']

            # adhoc_vector = obj['adhoc_vector']
            # static_vector = obj['static_vector']
            adhoc_vector = obj['text_vector']
            static_vector = obj['next_vector']

            # item_description = obj['item_description']
            item_description = obj['next_text']

            yield guid, {
                "recommendation_sid": recommendation_sid,
                "item_id": item_id,
                "inputs_embeds": adhoc_vector,
                "decoder_inputs_embeds": static_vector,
                "item_description": item_description,
            }
