import datasets
import json

from dataclasses import dataclass

from tunip.corpus_utils_v2 import CorpusToken
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
class DatasetUserQuery2ItemDescriptionConfig(datasets.BuilderConfig):

    dataset_path: str = f"{NAUTS_HOME}/data"
    dataset_file_suffix = TAGGED_JSONL_SUFFIX


class DatasetUserQuery2ItemDescription(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIG_CLASS = DatasetUserQuery2ItemDescriptionConfig

    BUILDER_CONFIGS = [
        DatasetUserQuery2ItemDescriptionConfig(
            name="Dataset for User Query and Item Description and Eval",
            version=datasets.Version("1.0.0"),
            description="Dataset for Item Description. Generation and Item Evalulation Regression",
        )
    ]

    def __init__(self, **kwargs):
        super(DatasetUserQuery2ItemDescription, self).__init__(**kwargs)

        self.logger = init_logging_handler_for_klass(klass=self.__class__)
        self.service_config = get_service_config()

        task_name = 'generation'
        domain_name = 'query2item_intro'

        snapshot_path_provider = SnapshotPathProvider(self.service_config)

        self.corpus_path_train_path = snapshot_path_provider.latest(
            f"/user/{self.service_config.username}/mart/corpus.tjsonl/{task_name}/{domain_name}/train"
        )
        self.corpus_path_dev_path = snapshot_path_provider.latest(
            f"/user/{self.service_config.username}/mart/corpus.tjsonl/{task_name}/{domain_name}/test"
        )
        self.corpus_path_test_path = snapshot_path_provider.latest(
            f"/user/{self.service_config.username}/mart/corpus.tjsonl/{task_name}/{domain_name}/test"
        )

        
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "recommendation_sid": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "next_text": datasets.Value("string"),
                    "evaluation": datasets.Value("float32"),
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

        spark = SparkConnector.getOrCreate(local=True)

        corpus_df = spark.read.json(corpus_path).select(
            "sid",
            "activity_tags_tokens",
            "user_query_tokens",
            "career_intro_tokens",
            "intro_tokens",
            "interaction",
            "passion",
            "promise",
        )

        if 'validation' in corpus_path or 'test' in corpus_path:
            corpus_df = corpus_df.sample(0.2, seed=42)
        rows = corpus_df.collect()
        self.logger.info("The number of examples from = %s", str(len(rows)))

        for guid, row in enumerate(corpus_df.collect()):
            obj = row.asDict()
            recommendation_sid = obj['sid']
            
            activity_tags_tokens = [CorpusToken.model_validate(t).surface for t in json.loads(obj['activity_tags_tokens'])]
            user_query_tokens = [CorpusToken.model_validate(t).surface for t in json.loads(obj['user_query_tokens'])]

            item_history_text_tokens = [CorpusToken.model_validate(t).surface for t in json.loads(obj['career_intro_tokens'])]
            item_description_tokens = [CorpusToken.model_validate(t).surface for t in json.loads(obj['intro_tokens'])]

            text = " ".join(activity_tags_tokens) + " " + " ".join(user_query_tokens)
            next_text = " ".join(item_history_text_tokens) + " " + " ".join(item_description_tokens)

            # evaluation = float(obj['evaluation'])
            interaction = float(obj['interaction'])
            passion = float(obj['passion'])
            promise = float(obj['promise'])

            evaluation = (interaction + passion + promise) / 3.

            yield guid, {
                "recommendation_sid": recommendation_sid,

                "text": text,
                "next_text": next_text,

                "evaluation": evaluation,
            }
