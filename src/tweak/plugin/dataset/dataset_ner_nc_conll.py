import datasets
import json
import logging
import pathlib

from dataclasses import dataclass
from typing import List

from tunip.corpus_utils import CorpusSeqLabel, CorpusToken
from tunip.env import NAUTS_HOME, TAGGED_JSONL_SUFFIX
from tunip.logger import init_logging_handler_for_klass


_CITATION = """\
"""

_DESCRIPTION = """\
"""


@dataclass
class DatasetNerNcConllConfig(datasets.BuilderConfig):

    dataset_path: str = f"{NAUTS_HOME}/data"
    dataset_file_suffix = TAGGED_JSONL_SUFFIX


class DatasetNerNcConll(datasets.GeneratorBasedBuilder):
    """SerpData dataset."""

    BUILDER_CONFIG_CLASS = DatasetNerNcConllConfig

    BUILDER_CONFIGS = [
        DatasetNerNcConllConfig(
            name="DatasetNerNcConll",
            version=datasets.Version("1.0.0"),
            description="Dataset for ner held by naver and changwon univ.",
        )
    ]

    def __init__(self, **kwargs):
        super(DatasetNerNcConll, self).__init__(**kwargs)

        self.logger = init_logging_handler_for_klass(klass=self.__class__)

        self.train_file = (
            pathlib.Path(self.config.dataset_path)
            / f"train_corpus{self.config.dataset_file_suffix}"
        )
        self.dev_file = (
            pathlib.Path(self.config.dataset_path)
            / f"dev_corpus{self.config.dataset_file_suffix}"
        )
        self.test_file = (
            pathlib.Path(self.config.dataset_path)
            / f"test_corpus{self.config.dataset_file_suffix}"
        )

        self.unk_file = open(
            pathlib.Path(self.config.dataset_path) / "unk_candidates.txt",
            mode="w+",
            encoding="utf-8",
        )

    def __del__(self):
        if self.unk_file.closed is False:
            self.unk_file.close()

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_starts": datasets.Sequence(datasets.Value("int32")),
                    "ner_ends": datasets.Sequence(datasets.Value("int32")),
                    "ner_tags": datasets.Sequence(datasets.Value("string"))
                }
            ),
            supervised_keys=None,
            homepage="https://www.aclweb.org/anthology/W03-0419/",
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
        logging.info("⏳ Generating examples from = %s", filepath)

        with open(filepath, encoding="utf-8") as f:
            guid = 0

            for line in f:
                guid = guid + 1

                obj = json.loads(line)
                tokens: List[CorpusToken] = [
                    CorpusToken.from_tuple_entry(t) for t in obj["tokens"]
                ]
                labels: List[CorpusSeqLabel] = [
                    CorpusSeqLabel.from_tuple_entry(l) for l in obj["labels"]
                ]

                # filtering tokens including space character
                new_tokens = []
                new_labels = []
                for i, t in enumerate(tokens):
                    if t.surface[0].isspace():  # to ignore some space characters like no break space
                        continue
                    else:
                        new_tokens.append(t)
                
                tokens = new_tokens

                head_tokens_updated, token_indices, surfaces_updated = CorpusToken.fit_into_split_words(tokens)

                # record to unk candidates
                if token_indices == [[0]]:
                    self.unk_file.write(surfaces_updated[0][:] + "\n")

                surfaces = [x.surface for x in head_tokens_updated]

                try:
                    iob_ner_tags = [entry.label for entry in labels]
                    ner_starts = [entry.start for entry in labels]
                    ner_ends = [entry.end for entry in labels]
                except ValueError as ve:
                    self.logger.error(
                        f"[ValueError]: text {obj['text']} and tokens {tokens}"
                    )
                    self.logger.error(f"{iob_ner_tags}")
                    continue
                except UserWarning as uw:
                    self.logger.warn(
                        f"[UserWarning]: text {obj['text']} and tokens {tokens}"
                    )
                    exit(0)

                yield guid, {
                    "id": str(guid),
                    "text": obj["text"],
                    "tokens": surfaces,
                    "ner_starts": ner_starts,
                    "ner_ends": ner_ends,
                    "ner_tags": iob_ner_tags,
                }
