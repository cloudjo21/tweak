import random
import timeit

from heapq import nlargest


def sample_iter(iterable, sample_size):
    results = []
    for i, v in enumerate(iterable):
        r = random.randint(0, i)
        if r < sample_size:
            if i < sample_size:
                results.insert(r, v) # add first sample_size items in random order
            else:
                results[r] = v # at a decreasing rate, replace random items

    if len(results) < sample_size:
        raise ValueError("Sample larger than population.")

    return results

def sample_iter_mid_size(iterable, sample_size):
    return (x for _, x in nlargest(sample_size, ((random.random(), x) for x in iterable)))

def sample_iter_lrg_size(iterable, sample_size):
    results = []
    iterator = iter(iterable)
    # Fill in the first sample_size elements:
    for _ in range(sample_size):
        results.append(next(iterator))
    random.shuffle(results)  # Randomize their positions
    for i, v in enumerate(iterator, sample_size):
        r = random.randint(0, i)
        if r < sample_size:
            results[r] = v  # at a decreasing rate, replace random items

    if len(results) < sample_size:
        raise ValueError("Sample larger than population.")
    return results

if __name__ == '__main__':
    pop_sizes = [int(10e+3),int(10e+4),int(10e+5),int(10e+5),int(10e+5),int(10e+5)*5]
    k_sizes = [int(10e+2),int(10e+3),int(10e+4),int(10e+4)*2,int(10e+4)*5,int(10e+5)*2]

    for pop_size, k_size in zip(pop_sizes, k_sizes):
        pop = range(pop_size)
        k = k_size
        t1 = timeit.Timer(stmt='sample_iter(pop, %i)'%(k_size), setup='from __main__ import sample_iter,pop')
        t2 = timeit.Timer(stmt='sample_iter_mid_size(pop, %i)'%(k_size), setup='from __main__ import sample_iter_mid_size,pop')
        t3 = timeit.Timer(stmt='sample_iter_lrg_size(pop, %i)'%(k_size), setup='from __main__ import sample_iter_lrg_size,pop')

        print('Sampling {0} from {1}'.format(k, pop_size))
        print ('Using sample_iter %1.4fs' % (t1.timeit(number=100) / 100.0))
        print ('Using sample_iter_mid_size %1.4fs' % (t2.timeit(number=100) / 100.0))
        print ('Using sample_iter_lrg_size %1.4fs' % (t3.timeit(number=100) / 100.0))