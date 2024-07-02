import numpy as np
import scipy.spatial.distance as distance

from abc import ABC
from itertools import combinations
from pydantic import BaseModel, validator
from typing import List

from . import DistanceCalcStatus


class JaccardDistanceCalcRequest(BaseModel):
    term_vector: np.ndarray
    num_rows: int

    @validator('term_vector', pre=True)
    def parse_term_vactor(v):
      return np.array(v, dtype=float)

    class Config:
        arbitrary_types_allowed = True


class JaccardDistanceCalcResponse(BaseModel):
    distances: List[np.float64]
    status: DistanceCalcStatus = DistanceCalcStatus.OK

    class Config:
        arbitrary_types_allowed = True


class JaccardDistanceCalcMethod(ABC):

    def status_by_num_rows(self, num_rows):
        if num_rows > 1:
            return DistanceCalcStatus.OK
        elif num_rows == 1:
            return DistanceCalcStatus.ONLY
        else: # num_rows == 0
            return DistanceCalcStatus.EMPTY


class JaccardDistanceCalcMethodA(JaccardDistanceCalcMethod):
    """
    scipy.dist.jaccard
    """
    def __call__(self, calc_request: JaccardDistanceCalcRequest):
        distances = []

        # constraint to minimum number of rows
        num_rows = min(calc_request.term_vector.shape[0], calc_request.num_rows)

        # make index combinations
        comb_indices = list(combinations(range(0, num_rows), 2))

        # calc_jaccard_distance_naive
        for i, j in comb_indices:
            # list of numpy.float64
            distances.append(distance.jaccard(calc_request.term_vector[i], calc_request.term_vector[j]))
        calc_response = JaccardDistanceCalcResponse(
            distances=distances,
            status=self.status_by_num_rows(num_rows)
        )
        return calc_response


class JaccardDistanceCalcMethodB(JaccardDistanceCalcMethod):
    """
    np.append.bitwise
    """
    def __call__(self, calc_request: JaccardDistanceCalcRequest):
        distances = []

        # constraint to minimum number of rows
        num_rows = min(calc_request.term_vector.shape[0], calc_request.num_rows)

        # make index combinations
        comb_indices = list(combinations(range(0, num_rows), 2))

        vecs_a, vecs_b = [], []
        for i, j in comb_indices:
            vecs_a.append(calc_request.term_vector[i])
            vecs_b.append(calc_request.term_vector[j])
        a = np.vstack(vecs_a).astype(np.int8)
        b = np.vstack(vecs_b).astype(np.int8)
        c = np.bitwise_xor(a ,b).sum(axis=-1)
        d = np.bitwise_and(a ,b).sum(axis=-1)
        distances = c / (c+d)
        calc_response = JaccardDistanceCalcResponse(
            distances=distances.tolist(),
            status=self.status_by_num_rows(num_rows)
        )
        return calc_response


class JaccardDistanceCalcMethodC(JaccardDistanceCalcMethod):
    """
    np.chunk.bitwise
    """
    def __call__(self, calc_request: JaccardDistanceCalcRequest):
        # TODO set by nnc deploy config
        chunk = 4

        # constraint to minimum number of rows
        num_rows = min(calc_request.term_vector.shape[0], calc_request.num_rows)

        # make index combinations
        comb_indices = list(combinations(range(0, num_rows), 2))

        batch_size = int(len(comb_indices) / chunk)
        chunk = 0 if batch_size < 1 else chunk
        rest_index = len(comb_indices) - batch_size * chunk

        a = np.zeros([len(comb_indices), calc_request.term_vector.shape[1]], dtype=np.int8)
        b = np.zeros([len(comb_indices), calc_request.term_vector.shape[1]], dtype=np.int8)
        a_indices = list(map(lambda x: x[0], comb_indices))
        b_indices = list(map(lambda x: x[1], comb_indices))

        for u in range(0, chunk):
            batch_indexes = list(range(batch_size*u, batch_size*(u+1)))
            batch_offset = batch_size*u

            a[batch_offset : batch_offset + len(batch_indexes)] = calc_request.term_vector[np.take(a_indices, batch_indexes, axis=0)]
            b[batch_offset : batch_offset + len(batch_indexes)] = calc_request.term_vector[np.take(b_indices, batch_indexes, axis=0)]

        if rest_index > 0:
            a_rest_indices = a_indices[-rest_index:]
            b_rest_indices = b_indices[-rest_index:]
            
            if len(a_rest_indices)> 0:
                a[-rest_index:] = calc_request.term_vector[a_rest_indices]
            if len(b_rest_indices)> 0:
                b[-rest_index:] = calc_request.term_vector[b_rest_indices]

        c = np.bitwise_xor(a, b).sum(axis=-1)
        d = np.bitwise_and(a, b).sum(axis=-1)

        distances = c / (c+d)

        calc_response = JaccardDistanceCalcResponse(
            distances=distances.tolist(),
            status=self.status_by_num_rows(num_rows)
        )

        return calc_response


class BigramJaccardDistanceCalc:

    def __init__(self):
        self.calc = JaccardDistanceCalcMethodC()

    def __call__(self, calc_request: JaccardDistanceCalcRequest):
        return self.calc(calc_request=calc_request)
