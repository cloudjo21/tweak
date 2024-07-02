import json
import os
import torch.nn as nn
import torch
import transformers
from typing import Callable, Optional, Union

from tokenizers import Tokenizer
from transformers import (
    AutoTokenizer,
    PretrainedConfig
)
from transformers.file_utils import (
    CONFIG_NAME,
    FLAX_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    TF2_WEIGHTS_NAME,
    WEIGHTS_NAME,
    # is_offline_mode,
    # is_remote_url
)

from tunip.logger import init_logging_handler

from tweak.model.model_attributer import ModelAttributer
from tweak.model.multitask.configuration_multitask import MultitaskConfig


logger = init_logging_handler(name=__name__)


class MultitaskModel(transformers.PreTrainedModel):
    """
    hard-parameter sharing multi-task model
    """

    config_class = MultitaskConfig

    def __init__(
        self,
        config: MultitaskConfig,
        encoder,
        task_models: dict,
        tokenizer: Tokenizer
    ):
        super().__init__(config)

        self.encoder = encoder
        self.task_models = nn.ModuleDict(task_models)
        self.tokenizer = tokenizer

    @classmethod
    def create(cls, model_name: str, model_types: dict, model_configs: dict):
        """
        :model_name:  pretrained model name
        """
        shared_encoder = None
        tokenizer = None
        task_models = dict()
        for task_name, model_type in model_types.items():
            model = model_type.from_pretrained(
                model_name, config=model_configs[task_name]
            )
            if shared_encoder is None:
                shared_encoder = ModelAttributer.get_model(model)

            task_models[task_name] = model
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    # cache_dir=model_args.cache_dir,
                    use_fast=True,
                    # add_prefix_space=False if 'roberta' not in model_nameelse True
                )
        mtl_config = MultitaskConfig(config=model_configs)
        return cls(
            config=mtl_config,
            encoder=shared_encoder,
            task_models=task_models,
            tokenizer=tokenizer
        )

    @classmethod
    def load_task_models(
        cls,
        model_name_dict: dict,
        model_types: dict,
        model_configs: dict
    ):
        shared_encoder = None
        tokenizer = None
        task_models = dict()
        for task_name, model_type in model_types.items():
            model = model_type.from_pretrained(
                model_name_dict[task_name], config=model_configs[task_name]
            )
            if shared_encoder is None:
                shared_encoder = ModelAttributer.get_model(model)

            task_models[task_name] = model
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name_dict[task_name],
                    # cache_dir=model_args.cache_dir,
                    use_fast=True,
                    # add_prefix_space=False if 'roberta' not in model_nameelse True
                )
        mtl_config = MultitaskConfig(config=model_configs)
        return cls(
            config=mtl_config,
            encoder=shared_encoder,
            task_models=task_models,
            tokenizer=tokenizer
        )
    
    def forward(self, task_name, **kwargs):
        return self.task_models[task_name](**kwargs)

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        save_config: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        **kwargs,
    ):
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return
        os.makedirs(save_directory, exist_ok=True)

        # Attach architecture to the config
        for name, model in self.task_models.items():
            self.config.config_dict[name].architectures = [model.__class__.__name__]

        # Save the config
        if save_config:
            self.config.save_pretrained(save_directory)

        # Save the model
        if state_dict is None:
            state_dict = dict()
            for name, model in self.task_models.items():
                state_dict.update({
                    name: model.state_dict()
                })

        # Handle the case where some state_dict keys shouldn't be saved
        if self._keys_to_ignore_on_save is not None:
            for name, model in self.task_models.items():
                state_dict_of_model = model.state_dict()
                state_dict_of_model = {k: v for k, v in state_dict_of_model.items() if k not in self._keys_to_ignore_on_save}
                state_dict[name] = state_dict_of_model

        # If we save using the predefined names, we can load using `from_pretrained`
        for name, model in self.task_models.items():
            output_model_file = os.path.join(save_directory + "/" + name, WEIGHTS_NAME)
            save_function(state_dict[name], output_model_file)

        logger.info(f"Model weights saved in {output_model_file}")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        mtl_config = kwargs.pop("config", None)
        state_dict = kwargs.pop("state_dict", None)
        cache_dir = kwargs.pop("cache_dir", None)
        from_tf = kwargs.pop("from_tf", False)
        from_flax = kwargs.pop("from_flax", False)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        mirror = kwargs.pop("mirror", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        _fast_init = kwargs.pop("_fast_init", True)

        user_agent = {"file_type": "model", "framework": "pytorch", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        # if is_offline_mode() and not local_files_only:
        if not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        # directory which have config.json
        if mtl_config is None:
            mtl_config = f"{pretrained_model_name_or_path}"

        with open(f"{mtl_config}/{CONFIG_NAME}", "r") as reader:
            text = reader.read()
        mtl_config_json = json.loads(text)

        # Load config if we don't provide a configuration
        if not isinstance(mtl_config, PretrainedConfig):
            config_path = mtl_config if mtl_config is not None else pretrained_model_name_or_path
            mtl_config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                **kwargs,
            )
        else:
            model_kwargs = kwargs
        
        mtl_config.name_or_path = pretrained_model_name_or_path
        model_name_or_path = dict()
        for name in mtl_config.task_names:
            model_name_or_path[name] = mtl_config[name].name_or_path

        model = cls.load_task_models(
            model_name_dict=model_name_or_path,
            model_types=MultitaskConfig.model_types_from_config(mtl_config),
            model_configs=mtl_config.config_dict,
        )

        return model

    def load_state_dict(self, checkpoint_file: Union[str, os.PathLike], task_name: str):
        """
        Reads a PyTorch checkpoint file, returning properly model for each task
        """
        return self.task_models[task_name].load_state_dict(checkpoint_file)
