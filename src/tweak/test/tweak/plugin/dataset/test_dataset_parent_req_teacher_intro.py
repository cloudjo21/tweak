import unittest
from datasets import load_dataset


class DatasetLoadTest(unittest.TestCase):
    def test_datset_ner_hdfs(self):

        dataset_path=f"/user/ed/mart/corpus.tjsonl/generation/query2item_intro/train/20230203_095349_803711"
        download_mode="force_redownload"

        loaded_datasets = load_dataset(
            f"src/tweak/plugin/dataset/dataset_query2item_intro.py",
            dataset_path=dataset_path,
            download_mode=download_mode,
        )
        print(len(loaded_datasets["train"]))
        print(len(loaded_datasets["test"]))
        # print(len(loaded_datasets["validation"]))

        assert len(loaded_datasets["train"]) > 0
