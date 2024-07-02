import copy
import json
import os
from typing import Any, Dict, Union

from transformers import (
    __version__,
    AutoConfig,
    PretrainedConfig
)
from transformers.file_utils import (
    CONFIG_NAME,
    # is_remote_url,
)

from tunip.logger import init_logging_handler
from tweak.task.task_set import TaskType


logger = init_logging_handler(name=__name__)


class MultitaskConfig(PretrainedConfig):
    def __init__(self, config=None, **kwargs):
    # def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config_dict = dict()
        if config:
            for name, conf in config.items():
                self.config_dict[name] = conf

    def __getitem__(self, task_name):
        return self.config_dict[task_name]
    
    @property
    def task_names(self):
        return self.config_dict.keys()

    def set_config(self, config_dict):
        self.config_dict = config_dict

    def copy_config(self):
        return copy.deepcopy(self.config_dict)

    # @property
    # def name_or_path(self):
    #     return super().name_or_path

    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):

        # for name, model_to_save in self.task_models.items():
        # Only save the model itself if we are using distributed training
        # model_to_save = unwrap_model(self)

        # save config file for multitask
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")
        os.makedirs(save_directory, exist_ok=True)
        output_config_file = os.path.join(save_directory, CONFIG_NAME)
        mtl_config = {"multitask": [name for name in self.config_dict]}
        with open(output_config_file, "w", encoding="utf-8") as writer:
            writer.write(json.dumps(mtl_config, indent=2, sort_keys=True) + "\n")
        logger.info(f"Configuration saved in {output_config_file}")

        for task_name in self.config_dict:

            save_task_directory = save_directory + "/" + task_name

            if os.path.isfile(save_task_directory):
                raise AssertionError(f"Provided path ({save_task_directory}) should be a directory, not a file")
            os.makedirs(save_task_directory, exist_ok=True)
            # If we save using the predefined names, we can load using `from_pretrained`
            output_config_file = os.path.join(save_task_directory, CONFIG_NAME)

            self._to_json_file_with_task(output_config_file, task_name=task_name, use_diff=False)
            logger.info(f"Configuration saved in {output_config_file}")

            if push_to_hub:
                url = super._push_to_hub(save_files=[output_config_file], **kwargs)
                logger.info(f"Configuration pushed to the hub in this commit: {url}")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        json_file = pretrained_model_name_or_path + "/" + CONFIG_NAME
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        config_dict = json.loads(text)
        task_names = config_dict['multitask']
        model_configs = dict()
        for name in task_names:
            model_configs[name] = AutoConfig.from_pretrained(
                pretrained_model_name_or_path + "/" + name
            )
        return cls(model_configs), kwargs

    def _to_json_file_with_task(self, json_file_path: Union[str, os.PathLike], task_name: str, use_diff: bool = True):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (:obj:`str` or :obj:`os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
            use_diff (:obj:`bool`, `optional`, defaults to :obj:`True`):
                If set to ``True``, only the difference between the config instance and the default
                ``PretrainedConfig()`` is serialized to JSON file.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self._to_json_string_with_task(task_name, use_diff=use_diff))

    def _to_json_string_with_task(self, task_name: str, use_diff: bool = True) -> str:
        """
        Serializes this instance to a JSON string.

        Args:
            use_diff (:obj:`bool`, `optional`, defaults to :obj:`True`):
                If set to ``True``, only the difference between the config instance and the default
                ``PretrainedConfig()`` is serialized to JSON string.

        Returns:
            :obj:`str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        # if use_diff is True:
        #     config_dict = self.to_diff_dict()
        # else:
        config_dict = self._to_dict_with_task(task_name)
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def _to_dict_with_task(self, task_name) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            :obj:`Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(
            json.loads(self.config_dict[task_name].to_json_string())
        )
        # if hasattr(self.__class__, "model_type"):
        #     output["model_type"] = self.__class__.model_type

        # Transformers version when serializing the model
        output["transformers_version"] = __version__

        return output

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            :obj:`Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        config_dict = dict()
        for name, config in self.config_dict.items():
            config_dict.update({
                name: json.loads(config.to_json_string())
            })

        output = copy.deepcopy(config_dict)
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type

        # Transformers version when serializing the model
        output["transformers_version"] = __version__

        return output

    @classmethod
    def model_types_from_config(cls, config):
        config_dict = config.copy_config()
        model_types = dict()
        for task_name, config in config_dict.items():
            # TODO change the getter for downstream model name w.r.t. model architecture
            downstream_model_name = config.architectures[0]
            model_type_klass = TaskType.model_name2model_type(
                downstream_model_name.upper()
            )
            model_types[task_name] = model_type_klass
        return model_types
