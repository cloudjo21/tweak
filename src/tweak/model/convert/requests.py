from typing import Optional
from pydantic import BaseModel


class Torch2OnnxRequest(BaseModel):
    model_type: str

    domain_name: str
    domain_snapshot: str
    task_name: str

    tokenizer_name: Optional[str]
    pt_model_name: Optional[str]
    checkpoint: Optional[str]

    max_length: int = 128
    lang: str = 'ko'


class Torch2TorchScriptRequest(BaseModel):
    model_type: str

    domain_name: Optional[str]
    domain_snapshot: Optional[str]
    task_name: Optional[str]

    tokenizer_name: Optional[str]
    pt_model_name: Optional[str]
    checkpoint: Optional[str]

    max_length: int = 128
    lang: str = 'ko'
