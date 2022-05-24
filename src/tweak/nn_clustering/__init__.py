from pydantic import BaseModel
from typing import Optional

import orjson


def orjson_dumps(v, *, default):
    # orjson.dumps returns bytes, to match standard json.dumps we need to decode
    return orjson.dumps(v, default=default).decode()


class Linkage(BaseModel):
  id: int
  nn_id: int
  dist: float
  n_clu: int
  cid: Optional[int]


# class LinkageLineage(BaseModel):
#   linkage: Linkage
#   last_cid: int

#   class Config:
#     json_loads = orjson.loads
#     json_dumps = orjson_dumps

