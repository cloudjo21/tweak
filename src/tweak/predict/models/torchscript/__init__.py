import pathlib
import torch

from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import (
    Seq2SeqLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.tokenization_utils_base import BatchEncoding

from tweak.predict.models import (
    ModelConfig,
    PredictableModel,
    PreTrainedModelConfig
)


class TorchScriptModel(PredictableModel):

    def __init__(self, config: ModelConfig):
        pass
        
    def infer(self) -> ModelOutput:
        pass


class TorchScriptModelForPreTrained(TorchScriptModel):
    def __init__(self, model_config: PreTrainedModelConfig):
        super().__init__(model_config)

        model_path = f"{model_config.model_path}"
        if '.pt' in pathlib.Path(model_config.model_path).suffix:
            model_path = f"{model_config.model_path}"
        else:
            model_path = f"{model_config.model_path}/model.pt"
        self.model = torch.jit.load(model_path)

        self.model.eval()
    
    def infer(self, encoded: BatchEncoding):
        with torch.no_grad():
            values = [encoded['input_ids'], encoded['attention_mask'], encoded['token_type_ids']]
            # values = [encoded['input_ids'], encoded['token_type_ids'], encoded['attention_mask']]
            predictions = self.model(
                *values
                # input_ids=encoded["input_ids"]
            )
            return predictions


class TorchScriptEncoderForPreTrained(TorchScriptModel):
    def __init__(self, model_config: PreTrainedModelConfig):
        super().__init__(model_config)

        assert model_config.encoder_only is True

        model_path = f"{model_config.model_path}"
        if '.pt' in pathlib.Path(model_config.model_path).suffix:
            model_path = f"{model_config.model_path}"
        else:
            model_path = f"{model_config.model_path}/model.pt"
        self.model = torch.jit.load(model_path)

        self.model.eval()
    
    def infer(self, encoded: BatchEncoding):
        with torch.no_grad():
            values = [encoded['input_ids'], encoded['attention_mask']]
            predictions = self.model(
                *values
            )
            return predictions


class TorchScriptModelForTokenClassification(TorchScriptModel):

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)

        # TODO
        model_path = f"{model_config.model_path}/../{model_config.checkpoint}/{model_config.task_name}" if model_config.checkpoint else model_config.model_path
        if '.pt' in pathlib.Path(model_config.model_path).suffix:
            model_path = f"{model_config.model_path}"
        else:
            model_path = f"{model_config.model_path}/torchscript/model.pt"
        self.model = torch.jit.load(model_path)

        self.model.eval()

    def infer(self, encoded: BatchEncoding) -> TokenClassifierOutput:
        with torch.no_grad():
            values = [encoded['input_ids'], encoded['attention_mask'], encoded['token_type_ids']]
            predictions: TokenClassifierOutput = self.model(
                *values
                # input_ids=encoded["input_ids"]
            )
            return predictions


class TorchScriptModelForSeq2SeqLM(TorchScriptModel):

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)

        # TODO
        model_path = f"{model_config.model_path}/../{model_config.checkpoint}/{model_config.task_name}" if model_config.checkpoint else model_config.model_path
        if '.pt' in pathlib.Path(model_config.model_path).suffix:
            model_path = f"{model_config.model_path}"
        else:
            model_path = f"{model_config.model_path}/torchscript/model.pt"
        self.model = torch.jit.load(model_path)

        self.model.eval()

    def infer(self, encoded: BatchEncoding) -> Seq2SeqLMOutput:
        with torch.no_grad():
            values = [encoded['input_ids'], encoded['attention_mask']]
            predictions: Seq2SeqLMOutput = self.model(
                *values
                # input_ids=encoded["input_ids"]
            )
            return predictions


class TorchScriptEncoderForSeq2SeqLM(TorchScriptModel):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)

        assert model_config.encoder_only is True

        model_path = f"{model_config.model_path}"
        if '.pt' in pathlib.Path(model_config.model_path).suffix:
            model_path = f"{model_config.model_path}"
        else:
            model_path = f"{model_config.model_path}/model.pt"
        self.model = torch.jit.load(model_path)

        self.model.eval()

    def infer(self, encoded: BatchEncoding) -> Seq2SeqLMOutput:
        with torch.no_grad():
            values = [encoded['input_ids'], encoded['attention_mask']]
            predictions: Seq2SeqLMOutput = self.model(
                *values
                # input_ids=encoded["input_ids"]
            )
            return predictions

