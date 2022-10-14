from itertools import chain

# from tweak.pretrain.pretrain_data_iterator import PretrainingDatasetFetcher


class PretrainingDataIterator:

    # def __init__(self, fetcher: PretrainingDatasetFetcher, num_epochs: int, current_epoch=0):
    def __init__(self, fetcher, num_epochs, current_epoch=0, use_cyclical_iter=False):
        self.fetcher = fetcher
        self.num_epochs = num_epochs

        self.current_epoch = current_epoch

        self.use_cyclical_iter = use_cyclical_iter
        self.iterator = self._make_iterator(self.current_epoch)

    
    def _make_iterator(self, epoch):
        for index in chain(*[self.fetcher.ready(epoch)]):
            yield self.fetcher.fetch_with(index)
    

    def __iter__(self):
        return self
    

    def __next__(self):
        try:
            example = next(self.iterator)
        except StopIteration:
            self.current_epoch += 1
            print(f"current_epoch: {self.current_epoch}/{self.num_epochs}")
            if (self.current_epoch == self.num_epochs) and (self.use_cyclical_iter is False):
                raise
            else:
                self.current_epoch = self.current_epoch % self.num_epochs
                self.iterator = self._make_iterator(self.current_epoch)
                example = next(self.iterator)
        return example


# for _ in range(10):
#     example = dataset_fetcher.fetch()
#     batch = tuple(t.to('cuda') for t in example)
#     print(batch)
#     dataset_fetcher.reload()

# dataset_fetcher.stop()

