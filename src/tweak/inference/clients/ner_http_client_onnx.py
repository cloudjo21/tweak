import numpy as np
import tritonclient.http as tritonhttpclient
from tritonclient.utils import np_to_triton_dtype 

triton_host = 'localhost'
triton_port = '31016'
# model_name = 'ner.python'
# model_name = 'ner.torchscript'
# model_name = 'ner.onnx'

# model_name = 'ner'
model_name = 'plm'

model_input0_name = 'input_ids'
model_input1_name = 'attention_mask'
model_input2_name = 'token_type_ids'
# model_output_name = 'predictions'
model_output_name = 'output_logits'

SEQ_LENGTH = 128

triton_client = tritonhttpclient.InferenceServerClient(f"{triton_host}:{triton_port}", verbose=True)


input_ids_np = np.concatenate((np.array([2, 12, 13, 14, 3]), np.full_like(np.arange(SEQ_LENGTH-5), 1))).astype('int64').reshape(1, SEQ_LENGTH)
attention_mask_np = np.concatenate((np.array([1, 1, 1, 1, 1]), np.full_like(np.arange(SEQ_LENGTH-5), 0))).astype('int64').reshape(1, SEQ_LENGTH)
token_types_np = np.concatenate((np.array([0, 1, 1, 1, 0]), np.full_like(np.arange(SEQ_LENGTH-5), 0))).astype('int64').reshape(1, SEQ_LENGTH)


# input_ids = tritonhttpclient.InferInput('input_ids', [1, SEQ_LENGTH], "INT_SEQ_LENGTH")
input_ids = tritonhttpclient.InferInput(model_input0_name, [1, SEQ_LENGTH], np_to_triton_dtype(input_ids_np.dtype))
attention_mask = tritonhttpclient.InferInput(model_input1_name, [1, SEQ_LENGTH], np_to_triton_dtype(attention_mask_np.dtype))
input_ = tritonhttpclient.InferInput(model_input2_name, [1, SEQ_LENGTH], np_to_triton_dtype(token_types_np.dtype))
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

print(results.as_numpy('output_logits').shape)