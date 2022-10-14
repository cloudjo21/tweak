import datasets

from dataclasses import dataclass

from codish.corpus.aspect.nikl import AspectIncludenessExample

from tunip.env import TAGGED_JSONL_SUFFIX
from tunip.path.mart import MartCorpusDomainTrainingStepPath
from tunip.service_config import get_service_config
from tunip.snapshot_utils import SnapshotPathProvider
from tunip.spark_utils import spark_conn

from tweak import LOGGER


_CITATION = """\
"""

_DESCRIPTION = """\
"""

_HOMEPAGE = """\
"""


@dataclass
class DatasetAspectNiklConfig(datasets.BuilderConfig):
    """BuilderConfig for Aspect Analysis Task by NIKL"""

    dataset_file_suffix = TAGGED_JSONL_SUFFIX


class DatasetAspectNikl(datasets.GeneratorBasedBuilder):
    """Aspect Analysis Task dataset."""

    BUILDER_CONFIG_CLASS = DatasetAspectNiklConfig

    BUILDER_CONFIGS = [
        DatasetAspectNiklConfig(
            name="DatasetAspectNikl",
            version=datasets.Version("1.0.0"),
            description="Dataset for Aspect Analysis",
        )
    ]

    def __init__(self, **kwargs):
        super(DatasetAspectNikl, self).__init__(**kwargs)
        self.logger = LOGGER

        task_name = 'aspect'
        domain_name = 'absa2022'
        service_config = get_service_config()

        snapshot_path_provider = SnapshotPathProvider(service_config)
        
        corpus_path_train_path = snapshot_path_provider.latest(
            MartCorpusDomainTrainingStepPath(
                service_config.username, task_name, domain_name, 'train'
        ))
        corpus_path_dev_path = snapshot_path_provider.latest(
            MartCorpusDomainTrainingStepPath(
                service_config.username, task_name, domain_name, 'dev'
        ))

        self.train_path = str(corpus_path_train_path)
        self.dev_path = str(corpus_path_dev_path)

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION, 
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "labels": datasets.Sequence(datasets.Value("string")),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        """Returns SplitGenerators."""

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"dataset_path": self.train_path},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"dataset_path": self.dev_path},
            ),
            # datasets.SplitGenerator(
            #     name=datasets.Split.TEST,
            #     gen_kwargs={"filepath": downloaded_files["test"]},
            # ),
        ]

    def _generate_examples(self, dataset_path):
        self.logger.info("⏳ Generating examples from = %s", dataset_path)

        spark = spark_conn.session

        guid = 0
        for row in spark.read.json(dataset_path).collect():
            guid = guid + 1
            example = AspectIncludenessExample(text=row.text, labels=row.labels)
            yield guid, example.dict()
