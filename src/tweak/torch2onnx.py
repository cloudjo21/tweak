from tunip.service_config import get_service_config

from tweak.model.convert.requests import Torch2OnnxRequest
from tweak.model.convert.services import TorchModelConvertService


service_config = get_service_config()

conv_req = Torch2OnnxRequest(
    model_type='hf.token_classification_model',
    domain_name='kb',
    domain_snapshot='20210910_124234_840184',
    task_name='ner',
    tokenizer_name='bert_word_piece/kowiki',
    checkpoint='checkpoint-200'
)

model_converter = TorchModelConvertService(service_config)
model_converter(conv_req)
