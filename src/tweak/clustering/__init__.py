import numpy as np

from pydantic import BaseModel
from typing import List, Optional

from tweak.clustering.distance import DistanceCalcStatus


class NncRequest(BaseModel):
    distances: List[np.float64]
    dist_calc_status: DistanceCalcStatus
    method: Optional[str] = 'complete'

    class Config:
        arbitrary_types_allowed = True
