import json
import os
import torch

import triton_python_backend_utils as pb_utils

from torch.utils.dlpack import from_dlpack, to_dlpack
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer


class TritonPythonModel:

    def initialize(self, args):

        self.model_config = json.loads(args['model_config'])

        output0_config = pb_utils.get_output_config_by_name(
            self.model_config, "predictions")

        # # Convert Triton types to numpy types
        # self.output0_dtype = pb_utils.triton_string_to_numpy(
        #     output0_config['data_type'])

        self.config = AutoConfig.from_pretrained('/models/ner/1/config.json')
        self.model = AutoModelForTokenClassification.from_pretrained(
            '/models/ner/1/',  # make sure that pytorch_model.bin is in this directory
            config=self.config
        )

    def execute(self, requests):

        responses = []

        for request in requests:
            input_ids = pb_utils.get_input_tensor_by_name(request, "input_ids")
            attention_mask = pb_utils.get_input_tensor_by_name(request, "attention_mask")
            input_ = pb_utils.get_input_tensor_by_name(request, "input")

            input_ids = from_dlpack(input_ids.to_dlpack())
            attention_mask = from_dlpack(attention_mask.to_dlpack())
            input_ = from_dlpack(input_.to_dlpack())

            output = self.model(input_ids, attention_mask, input_)

            # MUST TO CONVER WITH CONTIGUOUS TENSOR
            capsuled_preds = pb_utils.Tensor.to_dlpack(output.logits)
            out_preds = pb_utils.Tensor.from_dlpack(capsuled_preds)

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_preds])
            responses.append(inference_response)

        return responses
