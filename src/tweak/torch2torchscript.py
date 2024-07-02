import argparse

from tunip.service_config import get_service_config

from tweak.model.convert.requests import Torch2TorchScriptRequest
from tweak.model.convert.torchscript.service import TorchScriptModelConvertService


def run(args):
    service_config = get_service_config()

    conv_req = Torch2TorchScriptRequest(
        model_type=args.model_type,
        pt_model_name=args.pt_model_name,
        tokenizer_name=args.tokenizer_name,
        domain_name=args.domain_name,
        domain_snapshot=args.domain_snapshot,
        task_name=args.task_name,
        max_length=args.max_length,
        device=args.device,
        encoder_only=args.encoder_only,
        checkpoint=None
    )

    model_converter = TorchScriptModelConvertService(service_config)
    model_converter(conv_req)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert pre-training language model from torch to torchscript')
    parser.add_argument(
        '--model_type',
        help='model type [hf.plm_model | hf.token_classification | hf.sequence_classification]',
        required=True
    )
    parser.add_argument(
        '--pt_model_name',
        help='pt model name for mart/plm/models/[name]',
        required=True
    )
    parser.add_argument(
        '--checkpoint',
        help='checkpoint of pt model for mart/plm/models/[name]',
        required=False,
        default=None
    )
    parser.add_argument(
        '--tokenizer_name',
        help='tokenizer name for mart/tokenizers/[name]',
        required=True
    )
    parser.add_argument(
        '--domain_name',
        help='domain name',
        required=False
    )
    parser.add_argument(
        '--domain_snapshot',
        help='domain snapshot',
        required=False
    )
    parser.add_argument(
        '--task_name',
        help='task name',
        required=False
    )
    parser.add_argument(
        '--max_length',
        help='max_length of input sequence',
        required=False,
        default=128
    )
    parser.add_argument(
        '--device',
        help='device option [cuda | cpu]',
        required=False,
        default='cuda'
    )
    parser.add_argument(
        '--encoder_only',
        help='whether the encoder of FT-model would be exported or not',
        required=False,
        default=False
    )
    run(parser.parse_args())
