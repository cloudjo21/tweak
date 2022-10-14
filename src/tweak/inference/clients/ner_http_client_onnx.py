import numpy as np
import tritonclient.http as tritonhttpclient
from tritonclient.utils import np_to_triton_dtype 

triton_host = 'localhost'
triton_port = '31016'
# model_name = 'ner.python'
# model_name = 'ner.torchscript'
model_name = 'ner.onnx'
# model_output_name = 'predictions'
model_output_name = 'output_logits'


triton_client = tritonhttpclient.InferenceServerClient(f"{triton_host}:{triton_port}", verbose=True)


input_ids_np = np.concatenate((np.array([2, 12, 13, 14, 3]), np.full_like(np.arange(32-5), 1))).astype('int64').reshape(1, 32)
attention_mask_np = np.concatenate((np.array([1, 1, 1, 1, 1]), np.full_like(np.arange(32-5), 0))).astype('int64').reshape(1, 32)
token_types_np = np.concatenate((np.array([0, 1, 1, 1, 0]), np.full_like(np.arange(32-5), 0))).astype('int64').reshape(1, 32)


# input_ids = tritonhttpclient.InferInput('input_ids', [1, 32], "INT_32")
input_ids = tritonhttpclient.InferInput('input_ids', [1, 32], np_to_triton_dtype(input_ids_np.dtype))
attention_mask = tritonhttpclient.InferInput('attention_mask', [1, 32], np_to_triton_dtype(attention_mask_np.dtype))
input_ = tritonhttpclient.InferInput('input', [1, 32], np_to_triton_dtype(token_types_np.dtype))
output = tritonhttpclient.InferRequestedOutput(
    model_output_name,
    binary_data=True
    # binary_data=False  # verbose for result data from inference server
)

input_ids.set_data_from_numpy(input_ids_np, binary_data=True)
attention_mask.set_data_from_numpy(attention_mask_np, binary_data=True)
input_.set_data_from_numpy(token_types_np, binary_data=True)

results = triton_client.infer(
    model_name,
    inputs=[input_ids, attention_mask, input_],
    outputs=[output]
)

print("# infer_result: ")
print(results)
print(results.as_numpy('output_logits'))

print(np.argmax(results.as_numpy('output_logits'), axis=2))
