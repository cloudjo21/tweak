import numpy as np

from itertools import combinations
from pydantic import BaseModel, validator
from typing import List


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

    class Config:
        arbitrary_types_allowed = True


class BigramJaccardDistanceCalc:

    def __init__(self):
        pass

    def __call__(self, calc_request: JaccardDistanceCalcRequest):
        distances = []

        # constraint to minimum number of rows
        num_rows = min(calc_request.term_vector.shape[0], calc_request.num_rows)

        # make index combinations
        comb_indices = list(combinations(range(0, num_rows), 2))

        # TODO if we met the longer and wider matrix (e.g., (300~, 5000~)), 
        # use the following codes to avoid memory re-allocation
        # import scipy.spatial.distance as distance
        # # calc_jaccard_distance_naive
        # for i, j in comb_indices:
        #     # list of numpy.float64
        #     distances.append(distance.jaccard(calc_request.term_vector[i], calc_request.term_vector[j]))
        # calc_response = JaccardDistanceCalcResponse(distances=distances)

        # calc_jaccard_distance_naive
        vecs_a, vecs_b = [], []
        for i, j in comb_indices:
            vecs_a.append(calc_request.term_vector[i])
            vecs_b.append(calc_request.term_vector[j])
        a = np.vstack(vecs_a).astype(np.int8)
        b = np.vstack(vecs_b).astype(np.int8)
        c = np.bitwise_xor(a ,b).sum(axis=-1)
        d = np.bitwise_and(a ,b).sum(axis=-1)
        distances = c / (c+d)
        calc_response = JaccardDistanceCalcResponse(distances=distances.tolist())

        return calc_response
