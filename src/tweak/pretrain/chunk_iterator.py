import os
import subprocess

from collections.abc import Iterator
from functools import reduce
from itertools import chain, islice


def get_lines(filepath):
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            yield line


def get_line_count(filepath):
    wc_output = subprocess.check_output(["wc", "-l", filepath])
    line_count = int(wc_output.decode("utf-8").split()[0])
    return line_count


def get_size_of_files(files):
    total_raw_txt_size = 0
    for input_file in files:
        total_raw_txt_size += os.path.getsize(input_file)
    return total_raw_txt_size


def sum_of_linecounts(filepaths):
    return reduce(lambda a, b: a + b, [get_line_count(x) for x in filepaths])


def chain_of_files(filepaths):
    # lines of files
    return chain(*[get_lines(filepath) for filepath in filepaths])


def chunked_iterators(itr_data, len_data, shard_size):
    assert isinstance(itr_data, Iterator) is True
    for _ in range(0, len_data, shard_size):
        yield islice(itr_data, shard_size)
        # yield list(islice(itr_data, shard_size))


itr_x = iter(list(range(96)))
itr_y = chunked_iterators(itr_x, 96, 10)

# print(itr_y)
print(list(itr_y))


file_paths = []

sum_of_line_count = sum_of_linecounts(file_paths)
lines_of_files = chain_of_files(file_paths)

res = chunked_iterators(lines_of_files, sum_of_line_count, 200)

class CyclicIterator:
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self._iterator = self._make_iter()

    def __iter__(self):
        return self
    
    def _make_iter(self):

        # MAKE ITERATOR FOR SLICED ITERATORS
        # sum_of_line_count = sum_of_linecounts(self.file_paths)
        # lines_of_files = chain_of_files(file_paths)
        # res = chunked_iterators(lines_of_files, sum_of_line_count, 200)

        # MAKE ITERATOR FOR SIMPLE NUMBERS
        res = iter([1,2,3,4])
        for r in res:
            yield r

    def __next__(self):
        try:
            res = next(self._iterator)
        except StopIteration:
            print('?!')
            self._iterator = self._make_iter()
            res = next(self._iterator)
        return res


inf_iter = CyclicIterator(file_paths)
for _, r in enumerate(inf_iter):
    print(r)

exit(0)
# generator (generator for iterators)
print(res)

# iterator (iterator for doc)
iter0 = next(res)
print(iter0)

# item (doc)
item0 = next(iter0)
print(item0)

# print(next(res))
# print(list(next(res)))
# print(list(next(res)))
