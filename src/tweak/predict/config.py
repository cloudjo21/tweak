from pydantic import BaseModel
from typing import Optional

from tunip.orjson_utils import *


class TokenizerConfig(BaseModel):
    model_path: str
    max_length: int = 128
    task_name: Optional[str]
    path: Optional[str] = None
    allow_tags: Optional[list] = []

    class Config:
        json_loads = orjson.loads
        json_dumps = orjson_dumps

