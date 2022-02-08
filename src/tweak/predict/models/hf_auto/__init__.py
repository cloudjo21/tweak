import torch

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForTokenClassification
)
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import (
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.tokenization_utils_base import BatchEncoding

from tweak.predict.models import (
    ModelConfig,
    PredictableModel,
    PreTrainedModelConfig
)


class HFAutoModel(PredictableModel):

    def __init__(self, config: ModelConfig):
        # self.model_dir = f"{config.model_path}/{config.checkpoint}/{config.task_name}" if config.checkpoint else config.model_path
        # self.model_dir = f"{config.model_path}/{config.checkpoint}/{config.task_name}" if config.checkpoint else config.model_path

        self.auto_config = AutoConfig.from_pretrained(
            config.model_path, # finetuning_task=config.task_name
        )
        # self.pt_model_name = auto_config._name_or_path

        
    def infer(self) -> ModelOutput:
        pass


class HFAutoModelForPreTrained(HFAutoModel):
    def __init__(self, model_config: PreTrainedModelConfig):
        super().__init__(model_config)

        model_path = f"{model_config.model_path}"
        self.model = AutoModel.from_pretrained(
            model_path,
            from_tf=False,
            config=self.auto_config
        )
        self.model.eval()
    
    def infer(self, encoded: BatchEncoding):
        predictions = self.model(
            input_ids=encoded["input_ids"]
        )
        return predictions


class HFAutoModelForTokenClassification(HFAutoModel):

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)

        # TODO
        model_path = f"{model_config.model_path}/../{model_config.checkpoint}/{model_config.task_name}" if model_config.checkpoint else model_config.model_path
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            from_tf=False,
            config=self.auto_config,
        )
        self.model.eval()

    def infer(self, encoded: BatchEncoding) -> TokenClassifierOutput:

        predictions: TokenClassifierOutput = self.model(
            input_ids=encoded["input_ids"]
        )
        return predictions
