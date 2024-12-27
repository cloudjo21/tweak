from typing import Union

from tunip.dfs_utils import DfsDownloader
from tunip.service_config import ServiceLevelConfig

from tweak import LOGGER
from tweak.predict.config import TokenizerConfig
from tweak.predict.models import ModelConfig, PreTrainedModelConfig


class ResourceMaterializer:
    @classmethod
    def apply_for_tokenizer(
        cls, config: TokenizerConfig, service_config: ServiceLevelConfig
    ):
        dfs_downloader = DfsDownloader(service_config)

        dfs_downloader.download(path=f"{config.model_path}/config.json")
        config.model_path = f"{service_config.local_prefix}{config.model_path}"

        if config.path:
            dfs_downloader.download(path=config.path)
            config.path = f"{service_config.local_prefix}{config.path}"


    @classmethod
    def apply_for_hf_model(
        cls,
        config: Union[ModelConfig, PreTrainedModelConfig],
        service_config: ServiceLevelConfig,
    ):
        dfs_downloader = DfsDownloader(service_config)

        # download checkpoint-*/, [task]/
        downloaded = dfs_downloader.download(path=config.model_path)
        if downloaded:
            LOGGER.info(f"{__class__.__name__} downloaded MODEL: {config.model_path}")
        if config.checkpoint:
            dfs_downloader.download(path=f"{config.model_path}/../{config.checkpoint}")
        config.model_path = f"{service_config.local_prefix}{config.model_path}"

        if downloaded:
            LOGGER.info(f"{__class__.__name__} downloaded model_path in local FS: {config.model_path}")
