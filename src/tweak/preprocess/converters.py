"""
convert dataset by teh features from tokenizer
"""
from abc import ABC, abstractmethod
from collections import OrderedDict

from torch.utils.data.dataset import Dataset
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from tunip.gold import iob_from_labels_and_token_offsets

from tweak import DEFAULT_PADDING
from tweak.label.label_builder import LabelBuilderFactory
from tweak.task.task_set import InputColumnType, Task, TaskType


class Converter(ABC):
    def __init__(self, task):
        self.task = task

    @abstractmethod
    def convert(self, examples):
        pass

    def labelize(self, dataset: Dataset):
        label_builder = LabelBuilderFactory.create(self.task)
        self.label_list, self.label2id = label_builder(dataset)

    def label_set(self):
        return self.label_list


class NotSupportedTaskForConverter(RuntimeError):
    pass


class NeedToLabelizedException(RuntimeError):
    pass


class TokenizerBasedConverter(Converter):
    def __init__(self, task, tokenizer):
        super().__init__(task)
        self.tokenizer = tokenizer

    def labelize(self, dataset: Dataset):
        if self.task.use_vocab_label is True:
            unique_label_list = [b[1] for b in sorted([(v,k) for k, v in self.tokenizer.vocab.items()], key=lambda a: a[0], reverse=False)]
            self.label_list = unique_label_list
            self.label2id = self.tokenizer.vocab
        else:
            super().labelize(dataset)


class SingleInOutConverter(TokenizerBasedConverter):
    def __init__(self, task, tokenizer):
        super().__init__(task, tokenizer)

    def convert(self, examples):
        # TODO
        pass


class LabelOnlyConverter(TokenizerBasedConverter):
    def __init__(self, task, tokenizer):
        super().__init__(task, tokenizer)

    def convert(self, examples):

        if not self.label2id:
            raise NeedToLabelizedException()

        is_split_into_words = has_token_column = self.task.has_input_column_type(InputColumnType.TOKENS)
        if is_split_into_words:
            input_column = self.task.get_input_column_by(column_type=InputColumnType.TOKENS)
        else:
            input_column = self.task.get_input_column_by(column_type=InputColumnType.TEXT)

        features = self.tokenizer(
            examples[input_column.name],
            # examples[self.task.input_column_name],
            max_length=self.task.max_length,
            padding="max_length",
            truncation=True,
            # pad_to_max_length=False,  # True if TPU
            is_split_into_words=is_split_into_words,
        )
        labels = []
        if self.task.problem_type == 'multi_label_classification':
            num_labels = len(self.label2id)
            for labels_ in examples[self.task.label_column_name]:
                label_ids_for_example = []
                for label in labels_:
                    label_ids_for_example.append(self.label2id[label])
                labels_for_example = [1.0 if l in label_ids_for_example else 0.0 for l in range(num_labels)]

                labels.append(labels_for_example)
        else:
            for label in examples[self.task.label_column_name]:
                label_id = self.label2id[label]
                labels.append(label_id)
        features["labels"] = labels
        return features


class IobSequenceConverter(TokenizerBasedConverter):
    def __init__(self, task, tokenizer):
        super().__init__(task, tokenizer)

        # [CLS] [UNK] [SEP]
        self.invalid_input_ids = [
            tokenizer.cls_token,
            tokenizer.unk_token,
            tokenizer.sep_token,
        ]

    def convert(self, examples):

        if not self.label2id:
            raise NeedToLabelizedException()

        has_token_column = self.task.has_input_column_type(InputColumnType.TOKENS)
        # has_text_column = self.task.has_input_column_type(InputColumnType.TEXT)

        # assert has_token_column
        # assert has_token_column and has_text_column

        is_split_into_words = (
            True if has_token_column else False
            # True if self.task.input_column_type is InputColumnType.TOKENS else False
        )
        text_input_column = self.task.get_input_column_by(column_type=InputColumnType.TEXT)
        if has_token_column:
            tokens_input_column = self.task.get_input_column_by(column_type=InputColumnType.TOKENS)
            input_column = tokens_input_column
        else:
            input_column = text_input_column
        tokenized_inputs = self.tokenizer(
            examples[input_column.name],
            # examples[self.task.input_column_name],
            max_length=self.task.max_length,
            padding="max_length",
            truncation=True,
            # use this argument when the input from our dataset is lists of words (with a label for each word).
            is_split_into_words=is_split_into_words,
            return_offsets_mapping=True,
        )

        example_labels = examples[self.task.label_column_name]
        example_label_starts = examples["ner_starts"]
        example_label_ends = examples["ner_ends"]
        example_label_tuples = []
        for starts, ends, labels in zip(
            example_label_starts, example_label_ends, example_labels
        ):
            example_label_tuples.append(
                [[st, en, la] for st, en, la in zip(starts, ends, labels)]
            )
        # TODO read example_label_tuples from CorpusSeqLabel.from_columns(..)

        label_lists = []
        for i, (labels, offsets) in enumerate(
            zip(example_label_tuples, tokenized_inputs["offset_mapping"])
        ):
            word_ids = (
                tokenized_inputs.word_ids(batch_index=i)
                if is_split_into_words is True
                else None
            )
            text = examples[text_input_column.name][i]

            if word_ids:
                space_offsets = list(map(lambda t: t[1], filter(lambda t: t[0] == ' ', list(zip(text, list(range(len(text))))))))
                # get the only labels without token offsets mapping if return_offsets_mapping is False
                label_mappings = iob_from_labels_and_token_offsets(
                    labels=labels, token_offsets=offsets, word_ids=word_ids, space_offsets=space_offsets
                )

                label_ids = []
                for word_id in word_ids:
                    if word_id:
                        label_ids.append(self.label2id[label_mappings[word_id]])
                    else:
                        label_ids.append(-100)
            else:
                label_mappings = iob_from_labels_and_token_offsets(
                    labels=labels, token_offsets=offsets
                )
                valid_offsets = list(filter(lambda x: x[1]!=0, offsets))
                label_ids = [-100 for i in range(self.task.max_length)]
                for i in range(1, len(valid_offsets)+1):
                    label = label_mappings[i]
                    label_ids[i] = self.label2id[label]
                    
            label_lists.append(label_ids)

        tokenized_inputs["labels"] = label_lists
        return tokenized_inputs


class SpanOffsetsConverter(TokenizerBasedConverter):
    def __init__(self, task, tokenizer):
        super().__init__(task, tokenizer)

    def convert(self, examples):
        input_column_name = self.task.input_column_name
        label_column_name = self.task.label_column_name
        label_names = self.task.label_names

        # offsets
        start = list(map(lambda x: x[0], examples[label_column_name]))
        end = list(map(lambda x: x[1], examples[label_column_name]))

        tokenized_inputs = self.tokenizer(
            # query
            examples[input_column_name],
            max_length=self.task.max_length,
            padding="max_length",
            truncation=True,
            pad_to_max_length=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            # is_split_into_words=True
        )
        # start_positions
        tokenized_inputs[label_names[0]] = start
        # end_positions
        tokenized_inputs[label_names[1]] = end
        return tokenized_inputs


class NextTokenConverter(TokenizerBasedConverter):
    def __init__(self, task, tokenizer):
        super().__init__(task, tokenizer)

    def convert(self, examples):

        # if not self.label2id:
        #     raise NeedToLabelizedException()

        input_column = self.task.get_input_column_by(InputColumnType.TEXT)
        # input_column = self.task.get_input_column_by(InputColumnType.TEXT, column_name='text')
        label_column_name = self.task.label_column_name
        # label_names = self.task.label_names

        tokenized_inputs = self.tokenizer(
            # query
            examples[input_column.name],
            max_length=self.task.max_length,
            padding="max_length",
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            # is_split_into_words=True
        )
        next_tokenized_inputs = self.tokenizer(
            # query
            examples[label_column_name],
            max_length=self.task.max_length,
            padding="max_length",
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            # is_split_into_words=True
        )

        label_lists = []
        for input_ids in next_tokenized_inputs.input_ids:
            label_lists.append(input_ids)
        tokenized_inputs['labels'] = label_lists

        return tokenized_inputs


class NextConcatSequenceEmbedsConverter(TokenizerBasedConverter):
    def __init__(self, task, tokenizer):
        super().__init__(task, tokenizer)

    @staticmethod
    def _yield_after_concat_fields(examples, input_embed_fields):
        for fields in zip(*[examples[field] for field in input_embed_fields]):
            yield list(fields)

    def convert(self, examples):
        assert 'inputs_embeds' in examples
        assert 'decoder_inputs_embeds' in examples

        label_column_name = self.task.label_column_name
        next_tokenized_inputs = self.tokenizer(
            examples[label_column_name],
            max_length=self.task.max_length,
            padding=DEFAULT_PADDING,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            # is_split_into_words=True
        )

        label_lists = []
        for input_ids in next_tokenized_inputs.input_ids:
            label_lists.append(input_ids)
        examples['labels'] = label_lists

        return examples

    def labelize(self, dataset: Dataset):
        # dummy method
        pass


class ConcatSequenceEmbeds2TargetScoreConverter(Converter):
    def __init__(self, task):
        super().__init__(task)

    def convert(self, examples):
        assert 'inputs_embeds' in examples
        assert 'decoder_inputs_embeds' in examples
        assert self.task.label_column_name in examples

        return examples

    def labelize(self, dataset: Dataset):
        # dummy method
        pass

    def label_set(self):
        # No need to build label set by traversing dataset.
        # In the case of Inverse Cloze Task,
        # just use label_set from features(tokenized_inputs) of Converter.
        return None


class InverseClozeTaskConverter(TokenizerBasedConverter):
    def __init__(self, task, tokenizer):
        super().__init__(task, tokenizer)

    def convert(self, examples):
        # label_column_name
        # label_names = self.task.label_names

        has_token_column = self.task.has_input_column_type(InputColumnType.TOKENS)
        # TODO 
        # question_column = self.task.get_input_column_by(column_type=InputColumnType.TOKENS, column_name='question')
        # context_column = self.task.get_input_column_by(column_type=InputColumnType.TOKENS, column_name='context')
        question_column = self.task.get_input_column_by(column_type=InputColumnType.TEXT, column_name='question')
        context_column = self.task.get_input_column_by(column_type=InputColumnType.TEXT, column_name='context')

        tokenized_inputs = self.tokenizer(
            examples[question_column.name],
            max_length=self.task.max_length,
            padding="max_length",
            truncation=True,
            is_split_into_words=has_token_column,
        )
        c_tokenized_inputs = self.tokenizer(
            examples[context_column.name],
            max_length=self.task.max_length,
            padding="max_length",
            truncation=True,
            is_split_into_words=has_token_column,
        )

        tokenized_inputs['context_input_ids'] = c_tokenized_inputs['input_ids']
        tokenized_inputs['context_attention_mask'] = c_tokenized_inputs['attention_mask']
        tokenized_inputs['context_token_type_ids'] = c_tokenized_inputs['token_type_ids']

        # TODO set context input embeddings
        # tokenized_inputs['context_input_embeds'] = c_tokenized_inputs['input_embeds']

        positive_or_not = examples['has_answer']
        positive_labels = []
        negative_labels = []
        batch_size = len(examples[question_column.name])
        for i in range(batch_size):
            if positive_or_not[i]:
                positive_labels.append(i)
            #     negative_labels.append(-1)
            else:
                # TODO
                # raise NoAnswerExample()
                raise Exception('Exception for the example which has no answer!!')
                # positive_labels.append(-1)
            #     negative_labels.append(i)

        tokenized_inputs['positive_labels'] = positive_labels
        # tokenized_inputs['negative_labels'] = negative_labels

        return tokenized_inputs

    def labelize(self, dataset: Dataset):
        # dummy method
        pass

    def label_set(self):
        # No need to build label set by traversing dataset.
        # In the case of Inverse Cloze Task,
        # just use label_set from features(tokenized_inputs) of Converter.
        return None


class ConverterFactory:
    @classmethod
    def create(cls, task: Task) -> Converter:
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            task.pretrained_model_name,
            use_fast=True,
        )
        tokenizer.pad_token = task.pad_token if task.pad_token else tokenizer.pad_token
        has_inputs_embeds = task.has_input_column_type(InputColumnType.EMBEDS)
        if task.task_type == TaskType.TOKEN_CLASSIFICATION:
            return IobSequenceConverter(task, tokenizer)
        elif task.task_type == TaskType.SEQUENCE_CLASSIFICATION:
            return LabelOnlyConverter(task, tokenizer)
        elif task.task_type == TaskType.CAUSAL_LM:
            if has_inputs_embeds:
                return NextConcatSequenceEmbedsConverter(task, tokenizer)
            else:
                return NextTokenConverter(task, tokenizer)
        elif task.task_type == TaskType.SEQ2SEQ_LM:
            if has_inputs_embeds:
                return NextConcatSequenceEmbedsConverter(task, tokenizer)
            else:
                return NextTokenConverter(task, tokenizer)
        elif task.task_type == TaskType.QUESTION_ANSWERING:
            return SpanOffsetsConverter(task, tokenizer)
        elif task.task_type == TaskType.INVERSE_CLOZE_TASK:
            return InverseClozeTaskConverter(task, tokenizer)
        elif task.task_type == TaskType.REGRESSION:
            if has_inputs_embeds:
                return ConcatSequenceEmbeds2TargetScoreConverter(task)
            else:
                raise NotSupportedTaskForConverter()
        else:
            raise NotSupportedTaskForConverter()
