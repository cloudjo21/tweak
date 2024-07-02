import unittest

from tunip.task.task_def import TaskType
from tweak.task.task_set import InputColumn, InputColumnType, Task
from tweak.preprocess.converters import NextTokenConverter

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class ConvertersTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_next_token_converter(self):
        task = Task(
            task_name="generate",
            task_type=TaskType.CAUSAL_LM,
            dataset_name='',
            dataset_path='',
            pretrained_model_name="skt/kogpt2-base-v2",
            # pretrained_model_name="hyunwoongko/kobart",
            input_columns=[
                InputColumn(type_=InputColumnType.TEXT, name="text"),
                InputColumn(type_=InputColumnType.TEXT, name="next_text"),
            ],
            input_column_name="text",
            label_column_name="next_text",
            max_length=256
        )
        pretrained_model_name = task.pretrained_model_name
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            pretrained_model_name,
            use_fast=True,
            # bos_token='</s>', eos_token='</s>', unk_token='<unk>',
            # pad_token='<pad>', mask_token='<mask>',
        )
        tokenizer.pad_token = '<pad>'
        converter = NextTokenConverter(task, tokenizer)
        examples = None
        examples = {"text": ["안녕하세"], "next_text": ["녕하세요"]}
        tokenized_inputs = converter.convert(examples)
        assert tokenized_inputs
