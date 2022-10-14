import random

from abc import ABC
from enum import Enum
from itertools import dropwhile, filterfalse, repeat
from typing import List, Optional

from tunip.constants import SPECIAL_TOKENS_FOR_NSP, CLS, SEP

from tweak.pretrain.builders import PretrainingDocumentExampleBuilder
from tweak.pretrain.examples import (
    BertModelExample,
    MaskedLanguageModelExample,
    NextSentencePredictionExample,
    PretrainingTaskExample,
    PretrainingTaskExampleException
)
from tweak.pretrain.masked_language_model_example_builder import MaskedLanguageModelExampleBuilder
from tweak.pretrain.next_sentence_prediction_example_builder import NextSentencePredictionExampleBuilder, NextSentencePredictionExampleBuildRequest


class BertDocumentExampleBuilder(PretrainingDocumentExampleBuilder):
    """
      - pre-trains for document
      - includes NSP and MLM pre-training tasks
    """

    def __init__(self, documents, n_of_documents, max_length, max_predictions_per_sent, tokenizer, vocab_tokens):
        super().__init__()

        self.documents = documents
        self.n_of_documents = n_of_documents

        # Probability to create a sequence shorter than maximum sequence length
        self.short_seq_prob = 0.1
        self.rv = random.Random(42)

        # TODO get tokenizer and short_seq_prob from PretrainTask
        self.tokenizer = tokenizer
        self.vocabs = self.tokenizer.get_vocab()
        self.max_length = max_length
        self.pad_token_id = self.tokenizer.pad_token_id

        self.nsp_example_builder = NextSentencePredictionExampleBuilder(documents)
        self.mlm_example_builder = MaskedLanguageModelExampleBuilder(max_predictions_per_sent, vocab_tokens)
    

    def build(self, document, document_id, verbose) -> List[PretrainingTaskExample]:
        # document: sentences of tokens: List[List[str]]
        examples = []

        n_of_sent = len(document)
        max_n_tokens = self.max_length - len(SPECIAL_TOKENS_FOR_NSP)
        target_n_tokens = self.rv.randint(2, max_n_tokens) if self.rv.random() < self.short_seq_prob else max_n_tokens

        bunch_of_sents = []
        bunch_n_tokens = 0
        sent_index = 0
        for i, sent in enumerate(document):

            if len(sent) < 1:
                continue

            bunch_of_sents.append(sent)
            bunch_n_tokens += len(sent)

            try:
                training_example, sent_index = self._get_example_or_not(document, document_id, sent_index, bunch_of_sents, bunch_n_tokens, max_n_tokens, target_n_tokens)
            except PretrainingTaskExampleException as ptee:
                print(f"[WARNING] document_id: {document_id}, sent: {sent}")
                continue

            if verbose and (0 <= i < 2):
                print(f"[VERBOSE] {bunch_of_sents[:3]}")

            if training_example:
                examples.append(training_example)
                bunch_of_sents = []
                bunch_n_tokens = 0

        return examples


    def _get_example_or_not(self, document, document_id, sent_index, bunch_of_sents, bunch_n_tokens, max_n_tokens, target_n_tokens) -> (Optional[BertModelExample], Optional[int]):
        if bunch_of_sents and (sent_index == len(document) - 1 or bunch_n_tokens >= target_n_tokens):
            nsp_example: NextSentencePredictionExample = self.nsp_example_builder(
                NextSentencePredictionExampleBuildRequest(
                    document_id, sent_index, bunch_of_sents, max_n_tokens, target_n_tokens
                )) 
            mlm_example = self.mlm_example_builder(nsp_example.tokens)

            input_ids = self.tokenizer.convert_tokens_to_ids(nsp_example.tokens)
            attention_mask = [1] * len(input_ids)
            token_type_ids = nsp_example.segments
            masked_lm_labels = self._get_masked_lm_labels(mlm_example, len(input_ids))

            # padding
            while len(input_ids) < self.max_length:
                input_ids.append(self.pad_token_id)
                attention_mask.append(self.pad_token_id)
                token_type_ids.append(self.pad_token_id)
                masked_lm_labels.append(self.pad_token_id)

            example = BertModelExample(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                is_next_random=nsp_example.is_next_random,
                masked_lm_labels=masked_lm_labels
            )

            return example, self.nsp_example_builder.current_sent_index
        
        return None, sent_index+1
    
    def _get_masked_lm_labels(self, mlm_example: MaskedLanguageModelExample, len_input_ids: int):
        masked_lm_labels = [self.pad_token_id] * len_input_ids
        assert len(masked_lm_labels) == len_input_ids
        for t in mlm_example.tokens:
            masked_lm_labels[t.index] = self.vocabs[t.output_label] or self.tokenizer.unk_token_id
        return masked_lm_labels
