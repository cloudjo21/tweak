import queue
import threading


class ExampleSupplier(threading.Thread):
    def __init__(self, dataloaders, dataloader_indices):
        threading.Thread.__init__(self)
        self.in_queue = queue.Queue()
        self.out_queue = queue.Queue()
        self.dataloaders = dataloaders
        self.dataloader_indices = dataloader_indices
        self.cursor = min(3, len(dataloader_indices))
        for i in range(self.cursor):
            self.in_queue.put(self.dataloader_indices[i])

    def run(self):
        while True:
            dataset_index = self.in_queue.get(block=True)
            if dataset_index is None:
                break
            batch = next(self.dataloaders[dataset_index])
            self.in_queue.task_done()
            self.out_queue.put(batch)

    def get(self):
        batch = self.out_queue.get(timeout=2)
        self.out_queue.task_done()
        return batch

    def put(self):
        if self.cursor < len(self.dataloader_indices):
            self.in_queue.put(self.dataloader_indices[self.cursor])
            self.cursor += 1

    def stop(self):
        self.in_queue.put(None)
