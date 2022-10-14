import numpy as np


class StrIgnoreDevice(str):
    def to(self, device):
        return self


class TaskDataLoader:
    def __init__(self, task_name, data_loader):
        self.task_name = task_name
        self.data_loader = data_loader

        self.batch_size = data_loader.batch_size
        print(f"[TaskDataLoader.batch_size]: {self.batch_size}")
        self.dataset = data_loader.dataset

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        for batch in self.data_loader:
            batch["task_name"] = StrIgnoreDevice(self.task_name)
            yield batch


class MultitaskDataLoader:
    def __init__(self, data_loader_dict):
        self.data_loader_dict = data_loader_dict
        self.num_batches_dict = {
            task_name: len(data_loader)
            for task_name, data_loader in self.data_loader_dict.items()
        }
        self.task_name_list = list(self.data_loader_dict)
        self.dataset = [None] * sum(
            len(data_loader.dataset) for data_loader in self.data_loader_dict.values()
        )

    def __len__(self):
        return sum(self.num_batches_dict.values())

    def __iter__(self):
        """
        For each batch, sample a task, and yield a batch from the respective
        task Dataloader.
        """
        task_index_list = []
        for i, task_name in enumerate(self.task_name_list):
            task_index_list += [i] * self.num_batches_dict[task_name]

        task_index_list = np.array(task_index_list)
        np.random.shuffle(task_index_list)
        data_loader_iter_dict = {
            task_name: iter(data_loader)
            for task_name, data_loader in self.data_loader_dict.items()
        }
        for task_index in task_index_list:
            task_name = self.task_name_list[task_index]
            yield next(data_loader_iter_dict[task_name])
