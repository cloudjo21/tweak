import datasets
import json
import pathlib

from dataclasses import dataclass

from tunip.env import TAGGED_JSONL_SUFFIX
from tweak import LOGGER


_CITATION = """\
"""

_DESCRIPTION = """\
"""

_HOMEPAGE = """\
"""


@dataclass
class DatasetQaLocalConfig(datasets.BuilderConfig):
    """BuilderConfig for Question Answering"""

    dataset_file_suffix = TAGGED_JSONL_SUFFIX


class DatasetQaLocal(datasets.GeneratorBasedBuilder):
    """Question Answering dataset."""

    BUILDER_CONFIG_CLASS = DatasetQaLocalConfig

    BUILDER_CONFIGS = [
        DatasetQaLocalConfig(
            name="DatasetQaLocal",
            version=datasets.Version("1.0.0"),
            description="Dataset for Question Answering",
        )
    ]

    def __init__(self, **kwargs):
        super(DatasetQaLocal, self).__init__(**kwargs)
        self.logger = LOGGER
        # self.logger = init_logging_handler_for_klass(klass=self.__class__)
        self.logger.info(self.config.data_dir)

        self.train_file = (
            pathlib.Path(self.config.data_dir)
            / f"train_corpus{self.config.dataset_file_suffix}"
        )
        self.dev_file = (
            pathlib.Path(self.config.data_dir)
            / f"dev_corpus{self.config.dataset_file_suffix}"
        )
        self.test_file = (
            pathlib.Path(self.config.data_dir)
            / f"test_corpus{self.config.dataset_file_suffix}"
        )

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "doc_id": datasets.Value("string"),
                    "doc_title": datasets.Value("string"),
                    "context_id": datasets.Value("int32"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    # "question_tokens": datasets.Sequence(datasets.Value("string")),
                    "start": datasets.Value("int32"),
                    "end": datasets.Value("int32"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        """Returns SplitGenerators."""
        downloaded_files = {}

        downloaded_files["train"] = f"{self.train_file}"
        downloaded_files["dev"] = f"{self.dev_file}"
        downloaded_files["test"] = f"{self.test_file}"
        print("downloaded_files: ", downloaded_files)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": downloaded_files["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": downloaded_files["dev"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": downloaded_files["test"]},
            ),
        ]

    def _generate_examples(self, filepath):
        self.logger.info("⏳ Generating examples from = %s", filepath)

        with open(filepath, encoding="utf-8") as f:
            guid = 0

            for line in f:
                guid = guid + 1

                # TODO
                # obj = QAExample.parse_raw(line)
                obj = json.loads(line)

                yield guid, {
                    "id": str(guid),
                    "query": obj['query'],
                    # "query_tokens": obj['query_tokens'],
                    "start_position": obj['start'],
                    "end_position": obj['end'],
                }
