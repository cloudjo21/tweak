import datasets
import json
import logging
import pathlib

from dataclasses import dataclass
from typing import List

from tunip.env import NAUTS_HOME, TAGGED_JSONL_SUFFIX
from tunip.corpus_utils_v2 import CorpusRecord, CorpusSeqLabel, CorpusToken
# from tunip.logger import init_logging_handler_for_klass
from tweak import LOGGER


_CITATION = """\
"""

_DESCRIPTION = """\
"""


def get_just_labels(obj: CorpusRecord):
    has_boundary = True 
    label_entries = []
    start, end, label = None, None, None
    for obj_label in obj.labels:
        # obj_label.
        if obj_label.label.startswith('B-') and not has_boundary:
            label_entries.append(CorpusSeqLabel(start=start, end=end, label=label))
            start = obj_label.start
            end = obj_label.end
            label = obj_label.label[2:]
            has_boundary = False
        if obj_label.label.startswith('B-') and has_boundary:
            if label:
                label_entries.append(CorpusSeqLabel(start=start, end=end, label=label))
            start = obj_label.start
            end = obj_label.end
            label = obj_label.label[2:]
            has_boundary = False
        elif obj_label.label.startswith('I-'):
            end = obj_label.end
            has_boundary = True
    
    if has_boundary and label:
        label_entries.append(CorpusSeqLabel(start=start, end=end, label=label))
    return label_entries


@dataclass
class DatasetNerLabelWithIndexConfig(datasets.BuilderConfig):
    """BuilderConfig for SerpData"""

    # dataset_path: str = ""
    dataset_file_suffix = TAGGED_JSONL_SUFFIX


class DatasetNerLabelWithIndex(datasets.GeneratorBasedBuilder):
    """SerpData dataset."""

    BUILDER_CONFIG_CLASS = DatasetNerLabelWithIndexConfig

    BUILDER_CONFIGS = [
        DatasetNerLabelWithIndexConfig(
            name="DatasetNerLabelWithIndex",
            version=datasets.Version("1.0.0"),
            description="Dataset for ner with label index",
        )
    ]

    def __init__(self, **kwargs):
        super(DatasetNerLabelWithIndex, self).__init__(**kwargs)

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

    #     self.unk_file = open(
    #         pathlib.Path(self.config.dataset_path) / "unk_candidates.txt",
    #         mode="w+",
    #         encoding="utf-8",
    #     )

    # def __del__(self):
    #     if self.unk_file.closed is False:
    #         self.unk_file.close()

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
        self.logger.info("⏳ Generating examples from = %s", filepath)

        with open(filepath, encoding="utf-8") as f:
            guid = 0

            for line in f:
                guid = guid + 1

                obj = CorpusRecord.parse_raw(line)
                text = obj.text
                surfaces = [t.surface for t in obj.tokens]

                label_entries = get_just_labels(obj)
                iob_ner_tags = [t.label for t in label_entries]
                ner_starts = [t.start for t in label_entries]
                ner_ends = [t.end for t in label_entries]
                
                yield guid, {
                    "id": str(guid),
                    # "text": obj["text"],
                    "text": text,
                    "tokens": surfaces,
                    "ner_starts": ner_starts,
                    "ner_ends": ner_ends,
                    "ner_tags": iob_ner_tags,
                }
