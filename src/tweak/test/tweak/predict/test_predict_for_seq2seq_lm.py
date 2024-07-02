import unittest
import urllib.parse

from tunip.service_config import get_service_config
from tunip.path.mart import MartPretrainedModelPath
from tunip.path_utils import TaskPath

from tweak.predict.predict_seq2seq_lm import Seq2SeqLMPredictor
from tweak.predict.predictor import PredictorConfig
from tweak.predict.predictors import PredictorFactory


class PredictForSeq2SeqLMTest(unittest.TestCase):
    def setUp(self):
        service_config = get_service_config()
        self.model_name = "hyunwoongko/kobart"
        quoted_model_name = urllib.parse.quote(self.model_name, safe='')
        self.plm_model_path = MartPretrainedModelPath(
            user_name=service_config.username,
            model_name=quoted_model_name
        )
        self.tokenizer_path = str(self.plm_model_path) + "/vocab"

        # /data/home/ed/temp/user/ed/domains/query2item_intro/20230217_220119_168166
        self.task_path = TaskPath(service_config.username, 'query2item_intro', '20230328_130916_874499', 'generation')

        pred_config_json = {
            "predict_tokenizer_type": "auto",
            "predict_model_type": "auto",
            "predict_output_type": "hidden",
            "device": "cpu",
            "model_config": {
                "model_path": str(self.task_path),
                "model_name": self.model_name,
                "task_name": "generation",
                "task_type": "SEQ2SEQ_LM",
            },
            "tokenizer_config": {
                "model_path": str(self.plm_model_path),
                "path": self.tokenizer_path,
                "max_length": 128
            }
        }
        self.pred_config = PredictorConfig.model_validate(pred_config_json)

    def test_predict_for_hf_auto(self):
        predictor = PredictorFactory.create(self.pred_config)
        print(self.pred_config)
        res = predictor.predict(['영어 되시는 베이비시터 만 20세 이상 구합니다.'])
        # print('\n'.join(res))
        assert res is not None

    def test_predict_for_torchscript(self):
        pred_config_json = {
            "predict_tokenizer_type": "auto",
            "predict_model_type": "torchscript",
            "device": "cpu",
            "model_config": {
                "model_path": f"{str(self.task_path)}",
                "model_name": self.model_name,
                "task_name": "generate",
                "task_type": "SEQ2SEQ_LM",
            },
            "tokenizer_config": {
                "model_path": str(self.plm_model_path),
                "path": self.tokenizer_path,
                "max_length": 128
            }
        }
        self.pred_config = PredictorConfig.model_validate(pred_config_json)

        predictor = PredictorFactory.create(self.pred_config)
        print(self.pred_config)
        res = predictor.predict(['영어 되시는 베이비시터 만 20세 이상 구합니다.'])
        print('\n'.join(res))
        assert res is not None

    def test_predict_for_hf_auto_hidden(self):
        pred_config_json = {
            "predict_tokenizer_type": "auto",
            "predict_model_type": "auto",
            "predict_output_type": "hidden",
            "device": "cpu",
            "model_config": {
                "model_path": str(self.task_path),
                "model_name": self.model_name,
                "task_name": "generation",
                "task_type": "SEQ2SEQ_LM",
            },
            "tokenizer_config": {
                "model_path": str(self.plm_model_path),
                "path": self.tokenizer_path,
                "max_length": 128
            }
        }
        pred_config = PredictorConfig.model_validate(pred_config_json)

        predictor = PredictorFactory.create(pred_config)
        res = predictor.predict(['영어 되시는 베이비시터 만 20세 이상 구합니다.'])
        assert res is not None
        assert len(res.shape) > 1

    def test_predict_encoder_for_torchscript(self):
        d_model = 768
        pred_config_json = {
            "predict_tokenizer_type": "auto",
            "predict_model_type": "torchscript",
            "predict_output_type": "last_hidden.mean_pooling",
            "device": "cuda",
            "model_config": {
                "model_path": f"{str(self.task_path)}/torchscript/encoder",
                "model_name": self.model_name,
                "task_name": "generate",
                "task_type": "SEQ2SEQ_LM",
                "encoder_only": True
            },
            "tokenizer_config": {
                "model_path": str(self.plm_model_path),
                "path": self.tokenizer_path,
                "max_length": 128
            }
        }
        self.pred_config = PredictorConfig.model_validate(pred_config_json)

        predictor = PredictorFactory.create(self.pred_config)
        print(self.pred_config)
        res = predictor.predict(['영어 되시는 베이비시터 만 20세 이상 구합니다.'])
        print(res[0][:4])
        assert res is not None
        assert res.shape[1] == d_model

    def test_predict_encoder_for_torchscript_triton(self):
        d_model = 768
        pred_config_json = {
            "predict_tokenizer_type": "auto",
            "predict_model_type": "triton",
            # "predict_output_type": "last_hidden.mean_pooling",
            "predict_output_type": "last_hidden.global_mean_pooling",
            "device": "cpu",
            "model_config": {
                "model_path": f"{str(self.task_path)}",
                "model_name": self.model_name,
                "remote_backend": "torchscript",
                "task_name": "generation",
                "task_type": "SEQ2SEQ_LM",
                "encoder_only": True,
                "remote_host": "0.0.0.0",
                "remote_port": 31016,
                "remote_model_name": "generation.encoder"
            },
            "tokenizer_config": {
                "model_path": str(self.plm_model_path),
                "path": self.tokenizer_path,
                "max_length": 128
            }
        }
        self.pred_config = PredictorConfig.model_validate(pred_config_json)

        predictor = PredictorFactory.create(self.pred_config)
        print(self.pred_config)
        res = predictor.predict(['영어 되시는 베이비시터 만 20세 이상 구합니다.', '영어 되시는 베이비시터 만 20세 이상 구합니다.'])
        print(res.shape)
        print(res[0][:4])
        assert res is not None
        # assert res.shape[1] == d_model
        assert res.shape[1] == d_model and res.shape[0] == 1

    def test_predict_encoder_for_onnx_triton(self):
        d_model = 768
        pred_config_json = {
            "predict_tokenizer_type": "auto",
            "predict_model_type": "triton",
            "predict_output_type": "last_hidden.mean_pooling",
            "device": "cpu",
            "model_config": {
                "model_path": f"{str(self.task_path)}",
                # "model_path": f"{str(self.task_path)}/onnx/encoder",
                "model_name": self.model_name,
                "remote_backend": "onnx",
                "task_name": "generation",
                "task_type": "SEQ2SEQ_LM",
                "encoder_only": True,
                "remote_host": "0.0.0.0",
                "remote_port": 31016,
                "remote_model_name": "generation.encoder"
            },
            "tokenizer_config": {
                "model_path": str(self.plm_model_path),
                "path": self.tokenizer_path,
                "max_length": 128
            }
        }
        self.pred_config = PredictorConfig.model_validate(pred_config_json)

        predictor = PredictorFactory.create(self.pred_config)
        print(self.pred_config)
        res = predictor.predict(['영어 되시는 베이비시터 만 20세 이상 구합니다.'])
        print(res.shape)
        print(res[0][:4])
        assert res is not None
        assert res.shape[1] == d_model
