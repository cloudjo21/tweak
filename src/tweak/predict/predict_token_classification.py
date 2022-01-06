from pydantic import BaseModel
from torch import torch
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

from tweak.predict.predictor import Predictor


class PredictorConfig(BaseModel):
    # nugget/auto
    predict_tokenizer_type: str
    # triton/auto
    predict_model_type: str
    config: str


class TokenClassificationPredictor(Predictor):
    def __init__(
        self,
        # TODO alternate to PredictionToolbox
        predictor_config: PredictorConfig,
        max_length=32,
    ):
        super(TokenClassificationPredictor, self).__init__(predictor_config)

        self.max_length = max_length

        self.tokenzier = TokenizersFactory.create(
            predictor_config.predict_tokenizer_type,
            predictor_config.config
        )

        # model_dir = f"{model_dir}/{checkpoint}/{task_name}" if checkpoint else model_dir

        # self.config = AutoConfig.from_pretrained(
        #     model_dir, finetuning_task=task_name
        # )

        # pt_model_name = self.config._name_or_path

        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     pt_model_name,
        #     use_fast=True,
        #     add_prefix_space=False if "roberta" not in pt_model_name else True,
        # )

        self.model = ModelsFactory.create()

        self.model = AutoModelForTokenClassification.from_pretrained(
            model_dir if multi_task_or_not is True else model_dir,
            from_tf=False,
            config=self.config,
        )
        self.model.eval()

    def predict(self, texts: list, output_hidden_states=False, dim_argmax=2):
        """
        :param: texts        list of list => list of tokens, list => list of sentence
        """
        assert dim_argmax == 2
        is_split_into_words = is_split_into_words = (
            True if isinstance(texts, list) and isinstance(texts[0], list) else False
        )
        encoded = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            # pad_to_max_length=True,
            # return_special_tokens_mask=True,
            is_split_into_words=is_split_into_words,
        )
        output = self.model(input_ids=torch.tensor(encoded["input_ids"]))

        # Can be alternate to another PredictionBuildForXXX(IdWithLastHidden, Normal or Legacy ...)
        pred_id_list = torch.argmax(output.logits, dim=dim_argmax)
        num_tokens = len(pred_id_list)
        if output_hidden_states is True:
            result = {
                "prediction_ids": pred_id_list,
                "output_hidden_state_last": output.decoder_hidden_states[-1][
                    :, :num_tokens
                ],
            }
        else:
            result = {"prediction_ids": pred_id_list[:, :num_tokens]}

        return result
