from pydantic import (
    BaseModel,
    validator
)

from typing import Optional


class GenerationLogitsConfig(BaseModel):
    repetition_penalty: Optional[float]
    min_length: Optional[int]
    top_k: Optional[int]
    top_p: Optional[int]
    temperature: Optional[float]
    num_beams: Optional[int]

    forced_eos_token_id: Optional[int]  # TODO default value from model.config.forced_eos_token_id

    @validator('repetition_penalty')
    def check_repetition_penalty(cls, v):
        assert v <= 1.0 or (v is None)
        return v

    @validator('min_length')
    def check_min_length(cls, v):
        assert v >= 0 or (v is None)
        return v

    @validator('top_k')
    def check_top_k(cls, v):
        assert v > 0 or (v is None)
        return v
    
    @validator('top_p')
    def check_top_p(cls, v):
        assert v < 1.0 or (v is None)
        return v

    @validator('temperature')
    def check_temperature(cls, v):
        assert v != 1.0 or (v is None)
        return v

    @validator('num_beams')
    def check_num_beams(cls, v):
        assert v > 0 or (v is None)
        return v


class GenerationStopCriteriaConfig(BaseModel):
    max_length: Optional[int]
    max_time: Optional[int]

    @validator('max_length')
    def check_min_length(cls, v):
        assert v > 0 or (v is None)
        return v


class GenerationConfig(BaseModel):
    use_optimize: bool

    # False for CausalLM e.g., decoder-only model like GPT-2
    is_encoder_decoder: bool
    return_dict_in_generate: bool

    num_return_sequences: int

    # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int

    output_scores: bool
    output_attentions: bool
    output_hidden_states: bool
    use_cache: Optional[bool] = True

    logits_config: GenerationLogitsConfig
    stop_config: GenerationStopCriteriaConfig
