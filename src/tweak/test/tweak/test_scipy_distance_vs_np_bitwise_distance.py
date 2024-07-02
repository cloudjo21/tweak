import numpy as np
import scipy
import scipy.spatial.distance as distance
import timeit
from itertools import combinations

n_trial = 1
n_dim = 200
n_rows = 200

total_time = 0.0
term_vector = np.random.randint(0, 2, (n_rows, n_dim))
print(f"{term_vector.shape}")

comb_indices = list(combinations(range(0, n_rows), 2))

distances = []
for _ in range(0, n_trial):
    start = timeit.default_timer()

    # calc_jaccard_distance_naive
    for i, j in comb_indices:
        # list of numpy.float64
        distances.append(distance.jaccard(term_vector[i], term_vector[j]))

    total_time += (timeit.default_timer() - start)

print(f"total_time: {total_time} for scipy.distance")


total_time = 0.0
for _ in range(0, n_trial):
    start = timeit.default_timer()

    start0 = timeit.default_timer()
    c = np.zeros([len(comb_indices), n_dim])
    d = np.zeros([len(comb_indices), n_dim])
    r = 0
    vecs_a, vecs_b = [], []
    for i, j in comb_indices:
        c[r] = term_vector[i]
        d[r] = term_vector[j]
        r += 1
        # vecs_a.append(term_vector[i])
        # vecs_b.append(term_vector[j])
    
    c = c.astype(np.int8)
    d = d.astype(np.int8)

    # c = scipy.sparse.vstack(vecs_a).astype(np.int8)
    # d = scipy.sparse.vstack(vecs_b).astype(np.int8)
    # c = np.vstack(vecs_a).astype(np.int8)
    # d = np.vstack(vecs_b).astype(np.int8)
    # c = np.vstack(vecs_a)
    # d = np.vstack(vecs_b)
    print(f"vstack: took {timeit.default_timer() - start0}")

    start0 = timeit.default_timer()
    a = np.bitwise_xor(c ,d).sum(axis=-1)
    b = np.bitwise_and(c ,d).sum(axis=-1)
    print(f"bitwise: took {timeit.default_timer() - start0}")
    distances = a / (a+b)

    total_time += (timeit.default_timer() - start)

print(f"total_time: {total_time} for np.bitwise")


