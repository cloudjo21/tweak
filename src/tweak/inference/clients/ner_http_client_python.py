import numpy as np
import tritonclient.http as tritonhttpclient
from tritonclient.utils import np_to_triton_dtype 
from tritonclient.http import InferResult

triton_host = 'localhost'
# triton_port = '8000'
triton_port = '31016'
# model_name = 'ner.python'
model_name = 'ner'
# model_name = 'ner.torchscript'


model_input_0 = 'input_ids__0'
model_input_1 = 'attention_mask__1'
model_input_2 = 'token_type_ids__2'
# model_input_0 = 'INPUT__0'
# model_input_1 = 'INPUT__1'
# model_input_2 = 'INPUT__2'
# model_output_name = 'predictions'
model_output_name = 'logits__0'
# model_output_name = 'OUTPUT__0'

SEQ_LENGTH = 128

triton_client = tritonhttpclient.InferenceServerClient(f"{triton_host}:{triton_port}", verbose=True)

input_ids_np = np.concatenate((np.array([2, 12, 13, 14, 3]), np.full_like(np.arange(SEQ_LENGTH-5), 1))).astype('int32').reshape(1, SEQ_LENGTH)
attention_mask_np = np.concatenate((np.array([1, 1, 1, 1, 1]), np.full_like(np.arange(SEQ_LENGTH-5), 0))).astype('int32').reshape(1, SEQ_LENGTH)
token_types_np = np.concatenate((np.array([0, 1, 1, 1, 0]), np.full_like(np.arange(SEQ_LENGTH-5), 0))).astype('int32').reshape(1, SEQ_LENGTH)

# input_ids = tritonhttpclient.InferInput('input_ids', [1, 32], "INT_32")
input_ids = tritonhttpclient.InferInput(model_input_0, [1, SEQ_LENGTH], np_to_triton_dtype(input_ids_np.dtype))
attention_mask = tritonhttpclient.InferInput(model_input_1, [1, SEQ_LENGTH], np_to_triton_dtype(attention_mask_np.dtype))
token_type_ids = tritonhttpclient.InferInput(model_input_2, [1, SEQ_LENGTH], np_to_triton_dtype(token_types_np.dtype))
output = tritonhttpclient.InferRequestedOutput(
    model_output_name, binary_data=True
)

input_ids.set_data_from_numpy(input_ids_np, binary_data=True)
attention_mask.set_data_from_numpy(attention_mask_np, binary_data=True)
token_type_ids.set_data_from_numpy(token_types_np, binary_data=True)

result: InferResult = triton_client.infer(
    model_name,
    inputs=[
        input_ids,
        attention_mask,
        token_type_ids
    ],
    outputs=[output]
)

print(f"####\n{result}")
print(f"####\n{result.get_output(model_output_name)}")
print(f"####\n{result.as_numpy(model_output_name).shape}")
print(f"####\n{type(result.as_numpy(model_output_name))}")
print(f"####\n{result.as_numpy(model_output_name)}")
