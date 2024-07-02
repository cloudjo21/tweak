import argparse

from tunip.service_config import get_service_config

from tweak.model.convert.requests import Sklearn2OnnxRequest
from tweak.model.convert.sklearn.xgb import (
    SklearnModelConvertService
)


def run(args):
    service_config = get_service_config()

    conv_req = Sklearn2OnnxRequest(
        model_type=args.model_type,
        domain_name=args.domain_name,
        domain_snapshot=args.domain_snapshot,
        task_name=args.task_name,
    )

    model_converter = SklearnModelConvertService(service_config)
    model_converter(conv_req)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='convert scikit-learn model to onnx')
    parser.add_argument(
        '--model_type',
        help='model type [sklearn.classification_xgb | ...]',
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

    run(parser.parse_args())