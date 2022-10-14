import os

from abc import ABC

from torch.utils.data import Dataset

from tweak.pretrain.pretrain_data_packer import PretrainingDataPacker


class PretrainingDataset(ABC):
    pass


class BertPretrainingDataset(PretrainingDataset):

    def __init__(self, tokenizer, max_length, dataset_dir_path, epoch):
        dataset_filepath = self._random_dataset_file(dataset_dir_path, epoch)

        pretrain_data_packer = PretrainingDataPacker(tokenizer, max_length)
        self.dataset = pretrain_data_packer.load(dataset_filepath)

        self.unk_token_id = tokenizer.unk_token_id

        print(f"dataset file: [{dataset_filepath}] loaded.")

    
    def __getitem__(self, index):
        index_intra = index % len(self.dataset)
        example = self.dataset.examples[index_intra]

        # make example torch
        return example.torch(self.unk_token_id)
    

    def __len__(self):
        return len(self.dataset.examples)
    

    @staticmethod
    def _random_dataset_file(dataset_dir, index):
        files = [
            os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir)
        ]
        files = sorted(files)
        i = index % len(files)
        return files[i]
