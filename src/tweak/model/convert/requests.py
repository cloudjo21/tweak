from typing import Optional
from pydantic import BaseModel


class Torch2OnnxRequest(BaseModel):
    model_type: str

    domain_name: Optional[str]
    domain_snapshot: Optional[str]
    task_name: Optional[str]

    tokenizer_name: Optional[str]
    pt_model_name: Optional[str]
    checkpoint: Optional[str] = None

    max_length: int = 128
    lang: str = 'ko'
    device: str = 'cuda'

    encoder_only: bool = False


class Torch2TorchScriptRequest(BaseModel):
    model_type: str

    domain_name: Optional[str] = None
    domain_snapshot: Optional[str] = None
    task_name: Optional[str] = None

    tokenizer_name: Optional[str]
    pt_model_name: Optional[str]
    checkpoint: Optional[str] = None

    max_length: int = 128
    lang: str = 'ko'
    device: str = 'cuda'

    encoder_only: bool = False


class Sklearn2OnnxRequest(BaseModel):
    model_type: str
    domain_name: str
    domain_snapshot: str
    task_name: str
