# TODO ict dataset builder
# TODO ict resource builder
from typing import Dict

from transformers import (
    AutoConfig,
    AutoTokenizer
)
from transformers.configuration_utils import PretrainedConfig
from transformers.models.dpr.configuration_dpr import DPRConfig

from tweak.dataset.multitask_dataset import MultitaskDatasetBuilder
from tweak.dataset.resource_builder import ResourceBuilder
from tweak.preprocess.converters import Converter
from tweak.task.task_set import (
    AbstractTask,
    Task,
    TaskSet
)


# class InverseClozeTask


class InverseClozeTaskConfig(PretrainedConfig):
    ctx_encoder_config: DPRConfig
    question_encoder_config: DPRConfig


class InverseClozeTaskResourceBuilder(ResourceBuilder):
    
    def __init__(self, task_set: TaskSet):
        super().__init__(task_set)
    
    @property
    def model_configs(self):
        model_configs = dict()

        ict_task = self.task_set[0]
        ict_task.pretrained_model_name

        # ctx_config = DPRConfig(
        #     vocab_size
        # )
        plm_config = AutoConfig.from_pretrained(ict_task.pretrained_model_name)

        ict_config = DPRConfig(
            vocab_size=plm_config.vocab_size,

            hidden_size=plm_config.hidden_size
            num_hidden_layers=plm_config.hidden_size
        )

        model_configs[ict_task.task_name] = InverseClozeTaskConfig()
        return model_configs

    # def _get_model_config(self, task: AbstractTask, converter_set: Dict[str, Converter]) -> PretrainedConfig:
    #     return InverseClozeTaskConfig()
        # return super()._get_model_config(task, converter_set)
