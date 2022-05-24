import numpy as np
import scipy.spatial.distance as distance

from itertools import combinations
from pydantic import BaseModel
from typing import List


class JaccardDistanceCalcRequest(BaseModel):
    term_vector: np.ndarray
    num_rows: int


class JaccardDistanceCalcResponse(BaseModel):
    distances: List[np.float64]


class BigramJaccardDistanceCalc:

    def __init__(self):
        pass

    def __call__(self, calc_request: JaccardDistanceCalcRequest):
        distances = []

        # make index combinations
        comb_indices = list(combinations(range(0, calc_request.num_rows), 2))
        # calc_jaccard_distance_naive
        for i, j in comb_indices:
            # list of numpy.float64
            distances.append(distance.jaccard(calc_request.term_vector[i], calc_request.term_vector[j]))

        calc_response = JaccardDistanceCalcResponse(distances=distances)

        return calc_response
