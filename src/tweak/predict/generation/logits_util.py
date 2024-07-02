from pydantic import (
    BaseModel,
)
from transformers.generation.logits_process import (
    ForcedEOSTokenLogitsProcessor,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
    MaxLengthCriteria,
    MaxTimeCriteria,
)


class LogitsProcessorListFactory:

    @classmethod
    def create(self, req: BaseModel, eos_token_id: int, max_length: int):
        logits_procs = []
        if req.min_length:
            logits_procs.append(MinLengthLogitsProcessor(min_length=req.min_length, eos_token_id=eos_token_id))
        if req.repetition_penalty:
            logits_procs.append(RepetitionPenaltyLogitsProcessor(penalty=req.repetition_penalty))
        if req.forced_eos_token_id and max_length:
            logits_procs.append(ForcedEOSTokenLogitsProcessor(max_length=max_length, eos_token_id=req.forced_eos_token_id))
        return LogitsProcessorList(logits_procs)


class LogitsWarperListFactory:

    @classmethod
    def create(self, req: BaseModel):
        logits_procs = []

        if req.temperature:
            logits_procs.append(TemperatureLogitsWarper(req.temperature))
            print(f"append logit proc for temperature: {req.temperature}")
        if req.top_k:
            logits_procs.append(
                TopKLogitsWarper(req.top_k, min_tokens_to_keep=(2 if req.num_beams and req.num_beams > 1 else 1))
            )
            print(f"append logit proc for top-k: {req.top_k}")
        if req.top_p:
            logits_procs.append(
                TopPLogitsWarper(req.top_p, min_tokens_to_keep=(2 if req.num_beams and req.num_beams > 1 else 1))
            )
        return LogitsProcessorList(logits_procs)


class StoppingCriteriaListFactory:

    @classmethod
    def create(cls, req: BaseModel):
        criteria = []
        if req.max_length:
            criteria.append(MaxLengthCriteria(max_length=req.max_length))
        if req.max_time:
            criteria.append(MaxTimeCriteria(max_time=req.max_time))
        return StoppingCriteriaList(criteria)
