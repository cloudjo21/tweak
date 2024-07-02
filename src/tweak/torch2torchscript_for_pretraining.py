import argparse

from tunip.service_config import get_service_config

from tweak.model.convert.requests import Torch2TorchScriptRequest
from tweak.model.convert.torchscript.service import TorchScriptModelConvertService


def run(args):
    service_config = get_service_config()

    conv_req = Torch2TorchScriptRequest(
        model_type='hf.plm_model',
        pt_model_name=args.pt_model_name,
        tokenizer_name=args.tokenizer_name,
        encoder_only=args.encoder_only,
        device=args.device
    )

    model_converter = TorchScriptModelConvertService(service_config)
    model_converter(conv_req)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert pre-training language model from torch to torchscript')
    parser.add_argument(
        '--pt_model_name',
        help='pt model name for mart/plm/models/[name]',
        required=True
    )
    parser.add_argument(
        '--tokenizer_name',
        help='tokenizer name for mart/tokenizers/[name]',
        required=True
    )
    parser.add_argument(
        '--encoder_only',
        help='whether the encoder of FT-model would be exported or not',
        required=False,
        default=False
    )
    parser.add_argument(
        '--device',
        help='whether torch use cuda or cpu',
        required=False,
        default='cpu'
    )
    run(parser.parse_args())
