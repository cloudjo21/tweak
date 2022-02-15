from typing import Union

from tunip.file_utils import HttpBasedWebHdfsFileHandler
from tunip.service_config import ServiceLevelConfig

from tweak.predict.models import ModelConfig, PreTrainedModelConfig
from tweak.predict.config import TokenizerConfig


class ResourceMaterializer:

    @classmethod
    def apply_for_tokenizer(cls, config: TokenizerConfig, service_config: ServiceLevelConfig):
        webhdfs = HttpBasedWebHdfsFileHandler(service_config)
        count = 0
        print(count)
        count += 1

        cls._apply_if_service_from_hdfs(f"{config.model_path}/config.json", webhdfs, service_config)
        config.model_path = f"{service_config.local_prefix}/{config.model_path}"
        print(count)
        count += 1

        if config.path:
            cls._apply_if_service_from_hdfs(config.path, webhdfs, service_config)
            config.path = f"{service_config.local_prefix}/{config.path}"
        print(count)
        count += 1

    @classmethod
    def apply_for_hf_model(cls, config: Union[ModelConfig, PreTrainedModelConfig], service_config: ServiceLevelConfig):
        webhdfs = HttpBasedWebHdfsFileHandler(service_config)

        cls._apply_if_service_from_hdfs(config.model_path, webhdfs, service_config)
        config.model_path = f"{service_config.local_prefix}/{config.model_path}"


    @classmethod
    def _apply_if_service_from_hdfs(cls, path: str, file_handler: HttpBasedWebHdfsFileHandler, service_config: ServiceLevelConfig):
        print(path)
        if service_config.filesystem.upper() == 'HDFS':
            file_handler.download(path)
