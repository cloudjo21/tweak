import numpy as np

from transformers.tokenization_utils_base import BatchEncoding

import tritonclient.http as tritonhttpclient
from tritonclient.utils import np_to_triton_dtype

from tweak.predict.models import (
    ModelConfig,
    ModelOutput,
    PreTrainedModelConfig,
    PredictableModel,
)


class TritonClientModel(PredictableModel):

    DEFAULT_INPUT_SCHEMA = [
        ('input_ids', np.int64),
        ('attention_mask', np.int64),
        ('token_type_ids', np.int64),
    ]

    def __init__(self, model_config: ModelConfig):
        self.config = model_config

        assert self.config.remote_host
        assert self.config.remote_port
        assert self.config.remote_model_name

        self.triton_client = tritonhttpclient.InferenceServerClient(
            f"{model_config.remote_host}:{model_config.remote_port}",
            verbose=False
        )

    def _make_infer_input_array(self, encoded, input_schema_map: list):
        infer_input_array = []
        for key_name, value_type in input_schema_map:
            value_np = encoded[key_name].numpy().astype(value_type)
            infer_input = tritonhttpclient.InferInput(
                f"{key_name}",
                list(value_np.shape),
                np_to_triton_dtype(value_np.dtype)
            )
            infer_input.set_data_from_numpy(value_np)
            infer_input_array.append(infer_input)
        return infer_input_array

    def infer(self, encoded: BatchEncoding) -> ModelOutput:
        return self._infer_result(encoded, self.DEFAULT_INPUT_SCHEMA)
    
    def _infer_result(self, encoded: BatchEncoding, input_schema_map: list) -> ModelOutput:
        infer_input_array = self._make_infer_input_array(encoded, input_schema_map)

        logits_infer_req = tritonhttpclient.InferRequestedOutput(
            'output_logits', binary_data=True
        )

        result = self.triton_client.infer(
            self.config.remote_model_name,
            inputs=infer_input_array,
            outputs=[logits_infer_req]
        )
        return result.as_numpy('output_logits')


class TritonClientModelForPreTrainedModel(TritonClientModel):

    def __init__(self, model_config: PreTrainedModelConfig):
        super().__init__(model_config)

    def infer(self, encoded: BatchEncoding) -> ModelOutput:
        return super().infer(encoded)


class TritonClientModelForPreTrainedModelEncoder(TritonClientModel):
    PLM_ENCODER_INPUT_SCHEMA = [
        ('input_ids', np.int64),
        ('attention_mask', np.int64),
    ]

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)

    def infer(self, encoded: BatchEncoding) -> ModelOutput:
        return self._infer_result(encoded, self.PLM_ENCODER_INPUT_SCHEMA)


class TritonClientModelForTokenClassification(TritonClientModel):

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)

    def infer(self, encoded: BatchEncoding) -> ModelOutput:
        return super().infer(encoded)


class TritonClientModelForSeq2SeqLM(TritonClientModel):

    SEQ2SEQ_LM_INPUT_SCHEMA = [
        ('input_ids', np.int64),
        ('attention_mask', np.int64),
    ]

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)

    def infer(self, encoded: BatchEncoding) -> ModelOutput:
        return self._infer_result(encoded, self.SEQ2SEQ_LM_INPUT_SCHEMA)
