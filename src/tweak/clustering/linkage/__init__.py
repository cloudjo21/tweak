from pydantic import BaseModel
from typing import Optional


class Linkage(BaseModel):
  id: int
  nn_id: int
  dist: float
  n_clu: int
  cid: Optional[int]
