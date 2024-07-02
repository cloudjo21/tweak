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
                    "question": datasets.Value("string"),
                    "answers": datasets.Sequence(datasets.Value("string")),
                    "context": datasets.Value("string"),
                    "has_answer": datasets.Value("bool"),
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

                # TODO use codish example class
                # obj = AspectExample.parse_raw(line)

                try:
                    obj = json.loads(line)
                except json.decoder.JSONDecodeError as jde:
                    print(line)
                    print(str(jde))
                    continue

                # ignore the negative examples for Inverse Cloze Task Pretraining
                # if not obj['has_answer']:
                #     continue

                labels = list(set([a[0] for a in obj['annotation']]))

                yield guid, {
                    "labels": labels,
                    "text": obj['sentence_form'],
                    # "question": obj['question'],
                    # "answers": obj['answers'],
                    # "context": obj['text'],
                    # "has_answer": obj['has_answer'],
                }