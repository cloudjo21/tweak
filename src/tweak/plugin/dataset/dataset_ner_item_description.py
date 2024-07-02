import datasets
import json
import logging
import pathlib
import pandas as pd

from dataclasses import dataclass
from typing import List

from tunip.env import NAUTS_HOME, TAGGED_JSONL_SUFFIX
from tunip.corpus_utils import CorpusRecord, CorpusSeqLabel, CorpusToken
from tweak import LOGGER


_CITATION = """\
"""

_DESCRIPTION = """\
"""


@dataclass
class DatasetNerItemDescriptionConfig(datasets.BuilderConfig):
    dataset_path: str = f'{NAUTS_HOME}/data'
    dataset_file_suffix = '.jsonl'

class DatasetNerItemDescription(datasets.GeneratorBasedBuilder):
    """SerpData dataset"""

    BUILDER_CONFIG_CLASS = DatasetNerItemDescriptionConfig
    BUILDER_CONFIGS = [
        DatasetNerItemDescriptionConfig(
            name='DatasetNerItemDescription',
            version=datasets.Version('1.0.0'),
            description='Dataset for ner with label index'
        )
    ]


    def __init__(self, **kwargs):
        super(DatasetNerItemDescription, self).__init__(**kwargs)

        self.logger = LOGGER
        self.logger.info(self.config.data_dir)

        self.train_file = (
            pathlib.Path(self.config.data_dir)
            / f'train_corpus{self.config.dataset_file_suffix}'
        )
        self.dev_file = (
            pathlib.Path(self.config.data_dir)
            / f'dev_corpus{self.config.dataset_file_suffix}'
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
                    'id': datasets.Value('string'),
                    'text': datasets.Value('string'),
                    'tokens': datasets.Sequence(datasets.Value('string')),
                    'ner_starts': datasets.Sequence(datasets.Value('int32')),
                    'ner_ends': datasets.Sequence(datasets.Value('int32')),
                    'ner_tags': datasets.Sequence(datasets.Value('string'))
                }
            ),
            supervised_keys=None,
            homepage='https://www.aclweb.org/anthology/W03-0419/',
            citation=_CITATION
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        downloaded_files = {}

        downloaded_files['train'] = f'{self.train_file}'
        downloaded_files['dev'] = f'{self.dev_file}'
        downloaded_files['test'] = f'{self.test_file}'
        print('downloaded_files: ', downloaded_files)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={'filepath': downloaded_files['train']}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={'filepath': downloaded_files['dev']}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={'filepath': downloaded_files['test']}
            )
        ]

    def _generate_examples(self, filepath):
        self.logger.info(f'⏳ Generating examples from = {filepath}')

        with open(filepath, encoding="utf-8") as f:
            guid = 0

            for line in f:
                guid += 1

                obj = json.loads(line)
                tokens: List[CorpusToken] = [
                    CorpusToken(start = t[0],
                                end = t[1],
                                pos = t[2],
                                surface = t[3]) for t in obj["tokens"]
                ]
                labels: List[CorpusSeqLabel] = [
                    CorpusSeqLabel(start = int(l["start"]),
                                   end = int(l["end"]),
                                   label = l["label"]) for l in obj["labels"]
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