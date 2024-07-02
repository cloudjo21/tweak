from typing import Dict

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig

from tweak.preprocess.converters import Converter
from tweak.task.task_set import Task


def get_model_config(task: Task, converter_set: Dict[str, Converter]) -> PretrainedConfig:

    # in the case of downstream task
    model_config = AutoConfig.from_pretrained(
        task.pretrained_model_name,
        finetuning_task=task.task_name,
        num_labels=len(converter_set[task.task_name].label2id)
    )

    # in the case of pre-training task
    # TODO

    return model_config