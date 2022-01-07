from torch import torch
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer


class TokenClassificationPredictorLegacy(torch.nn.Module):
    def __init__(
        self,
        output_dir,
        task_name,
        checkpoint=None,
        multi_task_or_not=True,
        max_length=32,
    ):
        super(TokenClassificationPredictorLegacy, self).__init__()

        self.max_length = max_length

        output_dir = f"{output_dir}/{checkpoint}/{task_name}" if checkpoint else output_dir

        self.config = AutoConfig.from_pretrained(
            output_dir, finetuning_task=task_name
        )

        pt_model_name = self.config._name_or_path

        self.tokenizer = AutoTokenizer.from_pretrained(
            pt_model_name,
            use_fast=True,
            add_prefix_space=False if "roberta" not in pt_model_name else True,
        )

        self.model = AutoModelForTokenClassification.from_pretrained(
            output_dir if multi_task_or_not is True else output_dir,
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
