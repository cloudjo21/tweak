from tunip.service_config import ServiceLevelConfig

from tweak.model.convert.requests import Sklearn2OnnxRequest
from tweak.model.convert.sklearn import SklearnClassificationXgbModelConverter
from tweak.model.convert.sklearn.runtime_path import RuntimePathProviderFactory


class RuntimeModelConverterFactory:
    @classmethod
    def create(cls):
        pass


class SklearnModelConvertService:
    def __init__(self, service_config: ServiceLevelConfig):
        self.service_config = service_config

    def __call__(self, conv_req: Sklearn2OnnxRequest):
        runtime_path_provider = RuntimePathProviderFactory.create(
            convert_model_type=conv_req.model_type,
            user_name=self.service_config.username
        )
        model_path, runtime_model_path = runtime_path_provider.provide(conv_req)

        model_path = f"{self.service_config.local_prefix}{model_path}"
        runtime_model_path = f"{self.service_config.local_prefix}{runtime_model_path}"

        if conv_req.model_type == "sklearn.classification_xgb":
            model_converter = SklearnClassificationXgbModelConverter(
                model_path=model_path,
                runtime_model_path=runtime_model_path,
            )
        else:
            RuntimeError(f"Not supported model_type: {conv_req.model_type}")
        return model_converter()
