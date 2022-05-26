import argparse

from tunip.service_config import get_service_config

from tweak.model.convert.requests import Torch2OnnxForPretrainingRequest
from tweak.model.convert.services import TorchModelConvertService


service_config = get_service_config()

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str)
parser.add_argument("--pt_model_name", type=str)
parser.add_argument("--tokenizer_name", type=str, default=None)
args = parser.parse_args()


conv_req = Torch2OnnxForPretrainingRequest(
    model_type=args.model_type,
    pt_model_name=args.pt_model_name,
    tokenizer_name=args.tokenizer_name
)

model_converter = TorchModelConvertService(service_config)
model_converter(conv_req)
