import numpy as np
import scipy.spatial.distance as distance

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

        # make index combinations
        comb_indices = list(combinations(range(0, calc_request.num_rows), 2))
        # calc_jaccard_distance_naive
        for i, j in comb_indices:
            # list of numpy.float64
            distances.append(distance.jaccard(calc_request.term_vector[i], calc_request.term_vector[j]))

        calc_response = JaccardDistanceCalcResponse(distances=distances)

        return calc_response
