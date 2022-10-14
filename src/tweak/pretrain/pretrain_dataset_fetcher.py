import random

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler 
from torch.utils.data.sampler import RandomSampler

from tweak.pretrain.dataset import BertPretrainingDataset
from tweak.pretrain.example_supplier import ExampleSupplier


class PretrainingDatasetFetcher:

    def __init__(self, tokenizer, max_length, dataset_paths, train_batch_per_gpu=16, num_workers=8, local_rank=-1):
        self.dataloaders = dict()
        self.dataloader_indices = []

        self.tokenizer = tokenizer
        self.max_length = max_length

        self.dataset_paths = dataset_paths
        self.world_size = 1 if local_rank == -1 else torch.distributed.get_world_size()
        self.global_rank = 0
        # self.global_rank = torch.distributed.get_rank()
        # assert global_rank == 0
        self.train_batch_per_gpu = train_batch_per_gpu
        self.local_rank = local_rank
        self.num_workers = num_workers

        self.gradient_accum_step = 1  # assume that #-of-batch-per-gpu is less than total-batch-size

    # TODO PretrainingDataIterator
    # iteration for fixed number of epochs
    
    def ready(self, epoch):
        dataset_indices = []
        for i, dataset_path in enumerate(self.dataset_paths):
            dataset = BertPretrainingDataset(self.tokenizer, self.max_length, dataset_path, epoch)
            
            if len(dataset) < self.world_size * self.train_batch_per_gpu * self.gradient_accum_step:
                effective_batchsize = len(dataset)
            else:
                effective_batchsize = self._get_effective_batchsize(len(dataset))

            dataset_indices.extend([i] * effective_batchsize)
            self.dataloaders[i] = self._get_dataloader(dataset)
        
        random.shuffle(dataset_indices)
        
        self.dataloader_indices = []
        for index in dataset_indices:
            self.dataloader_indices.extend([index] * self.gradient_accum_step)
        
        # self.example_supplier = ExampleSupplier(self.dataloaders, self.dataloader_indices)
        # self.example_supplier.start()
    
        return self.dataloader_indices


    def fetch(self):
        # return self.example_supplier.get()
        pass

    def fetch_with(self, index):
        try:
            data = next(self.dataloaders[index])
            return data
        except ConnectionResetError as cre:
            print(str(cre))
            return None
    

    def reload(self):
        # self.example_supplier.put()
        pass
        

    def stop(self):
        # self.example_supplier.stop()
        pass


    def _get_sampler(self, dataset):
        return RandomSampler(dataset) if self.local_rank else DistributedSampler(dataset)

    def _get_dataloader(self, dataset: Dataset):
        # TODO adapt multistream dataloader using iterable multistream dataset
        sampler = self._get_sampler(dataset)
        return (x for x in DataLoader(dataset, batch_size=self.train_batch_per_gpu, sampler=sampler, num_workers=self.num_workers))

    def _get_effective_batchsize(self, length):
        return length // self.world_size // self.train_batch_per_gpu // self.gradient_accum_step
