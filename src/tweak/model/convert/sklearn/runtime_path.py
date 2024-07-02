from abc import ABC

from tunip.path_utils import TaskPath
from tweak.model.convert.requests import Sklearn2OnnxRequest


class RuntimePathProvider(ABC):
    pass


class RuntimePathProviderForClassificationXgb:
    def __init__(self, user_name):
        self.user_name = user_name
    
    def provide(self, conv_req: Sklearn2OnnxRequest):
        model_path = TaskPath(
            user_name=self.user_name,
            domain_name=conv_req.domain_name,
            snapshot_dt=conv_req.domain_snapshot,
            task_name=conv_req.task_name,
        )
        runtime_model_path = f"{str(model_path)}/onnx"
        return (model_path, runtime_model_path)


class RuntimePathProviderFactory:
    @classmethod
    def create(self, convert_model_type, user_name):
        if convert_model_type == 'sklearn.classification_xgb':
            return RuntimePathProviderForClassificationXgb(user_name)
        else:
            raise RuntimeError("Not Supported Model Type")
