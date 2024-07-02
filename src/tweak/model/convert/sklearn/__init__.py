import joblib
import orjson
import pathlib
import xgboost

from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from xgboost import XGBClassifier

from tweak import LOGGER
from tweak.model.convert import prepare_device

MODEL_NAME = "my-xgb"

class SklearnModelConverter:
    def __init__(self, model_path, runtime_model_path):
        self.model_path = model_path
        self.runtime_model_path = runtime_model_path


class SklearnClassificationXgbModelConverter(SklearnModelConverter):

    CONFIG_PB_TXT = """
platform: "onnxruntime_onnx"
max_batch_size: 8
input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [ %(FEATURE_COUNT)s ]
  }
]
output [
  {
    name: "label"
    data_type: TYPE_INT64
    dims: [ 1 ]
    reshape: { shape: [ ] }
  },
  {
    name: "probabilities"
    data_type: TYPE_FP32
    dims: [ 2 ]
  }
]
    """

    def __init__(self, model_path, runtime_model_path, device=None, feature_count=-1):
        super(SklearnClassificationXgbModelConverter, self).__init__(model_path, runtime_model_path)

        # TODO n_gpu_req option
        if not device:
            self.device, _ = prepare_device(n_gpu_req=-1)
        else:
            self.device = device
        
        if feature_count < 0:
            feature_name_map_filepath = f"{self.model_path}/feature_name_map.json"
            with open(feature_name_map_filepath, "r") as json_file:
                feature_name_map = orjson.loads(json_file.read())
            self.feature_count = len(feature_name_map)
        else:
            assert feature_count > 0
            self.feature_count = feature_count
        
    def __call__(self):
        try:
            runtime_model_filepath = f"{self.runtime_model_path}/model.onnx"
            pathlib.Path(self.runtime_model_path).mkdir(parents=True, exist_ok=True)

            model_filepath = f"{self.model_path}/model.ubj"
            model: xgboost.sklearn.XGBClassifier = joblib.load(model_filepath)

            update_registered_converter(
                XGBClassifier, 'XGBoostXGBClassifier',
                calculate_linear_classifier_output_shapes, convert_xgboost,
                options={'nocl': [True, False], 'zipmap': [False, False, 'columns']})

            model_onnx = convert_sklearn(
                model, MODEL_NAME, [('input', FloatTensorType([None, len(model.feature_names_in_)]))], 
                target_opset={'': 12, 'ai.onnx.ml': 2},
                options={id(model):{'zipmap': False}}  # option to deactivate dictionary type for probability per class
            )
            # model_onnx = convert_sklearn(model, MODEL_NAME, [('input', FloatTensorType([None, len(model.feature_names_in_)]))], target_opset={'': 12, 'ai.onnx.ml': 2})

            with open(runtime_model_filepath, "wb") as f:
                f.write(model_onnx.SerializeToString())

            # write down config.pbtxt
            LOGGER.info(f"write down config.pbtxt to {self.runtime_model_path}")
            runtime_config_filepath = f"{self.runtime_model_path}/config.pbtxt"
            config_pb_txt = str(self.CONFIG_PB_TXT % {"FEATURE_COUNT": str(self.feature_count)})
            with open(runtime_config_filepath, "w") as f:
                f.write(config_pb_txt)

        except Exception as e:
            LOGGER.error(e)
            return 500
        finally:
            return 0
