import argparse
import json
import os
import subprocess
import time

from functools import reduce
from itertools import chain, islice
from pathlib import Path

from tokenizers import SentencePieceBPETokenizer, BertWordPieceTokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from tunip.corpus.corpus_path_provider import CorpusPathProviderFactory
from tunip.env import NAUTS_LOCAL_ROOT
from tunip.path.mart import MartTokenizerPath
from tunip.preprocess import preprocess_korean
from tunip.service_config import get_service_config

def list_of_files(corpus_bin_paths):
    files = []
    for dir_path in corpus_bin_paths:
        path_provider = CorpusPathProviderFactory.create(
            corpus_type="dir",
            input_dir=dir_path,
            extensions=[".txt"],
            split_path=""
        )
        files.extend([str(p) for p in path_provider()])
    return files


def get_lines(filepath):
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            yield line


def chain_of_files(filepaths):
    # lines of files
    return chain(*[get_lines(filepath) for filepath in filepaths])


def preprocess(lines):
    for line in lines:
        # filter korean strictly
        yield preprocess_korean(line, strict=True)


def train_bert_wordpiece_tokenizer(files):
    tokenizer = BertWordPieceTokenizer(
        # vocab_file=None,
        clean_text=True,
        handle_chinese_chars=True,
        strip_accents=False, # Must be False if cased model
        lowercase=False,
        wordpieces_prefix="##"
    )
    tokenizer.train(files, limit_alphabet=50, vocab_size=35000)

    return tokenizer


def train_iter_bert_wordpiece_tokenizer(files):
    rows = preprocess(chain_of_files(files))

    tokenizer = BertWordPieceTokenizer(
        # vocab_file=None,
        clean_text=True,
        handle_chinese_chars=True,
        strip_accents=False, # Must be False if cased model
        lowercase=False,
        wordpieces_prefix="##"
    )
    tokenizer.train_from_iterator(rows, vocab_size=35000, min_frequency=4, show_progress=True, limit_alphabet=5000)

    return tokenizer


def train_iter_sentence_piece_tokenizer(files):
    rows = chain_of_files(files)

    tokenizer = SentencePieceBPETokenizer()
    # tokenizer.train(files, vocab_size=35000, min_frequency=5, show_progress=True, limit_alphabet=50)
    tokenizer.train_from_iterator(rows, vocab_size=35000, min_frequency=5, show_progress=True, limit_alphabet=50)

    return tokenizer


def train_tokenizer(tokenizer_name, files):
    if tokenizer_name == 'bert_word_piece':
        return train_iter_bert_wordpiece_tokenizer(files)
    elif tokenizer_name == 'sentence_piece':
        return train_iter_sentence_piece_tokenizer(files)


def main(args):

    files = list_of_files(args.corpus_paths)

    service_config = get_service_config(force_service_level='dev')

    tokenizer_dir_path = args.output_dir if args.output_dir else f"{NAUTS_LOCAL_ROOT}{str(MartTokenizerPath(user_name=service_config.username, tokenizer_name=args.tokenizer))}"
    print(f"Tokenizer would be written to this path: {str(tokenizer_dir_path)}")
    print(f"files: {', '.join(files)}")

    t = time.process_time()
    print("START TO TRAIN TOKENIZER", flush=True)
    tokenizer = train_tokenizer(args.tokenizer, files)
    print(f"END TO TRAIN TOKENIZER {(time.process_time()-t)/60.} minutes.")

    Path(tokenizer_dir_path).mkdir(parents=True, exist_ok=True)
    tokenizer_json_path = Path(f'{tokenizer_dir_path}/tokenizer.json')
    tokenizer.save(str(tokenizer_json_path))

    transformer_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_json_path)
    )
    transformer_tokenizer.save_pretrained(tokenizer_dir_path, legacy_format=False)
    result = transformer_tokenizer.tokenize('안녕 토크나이저!')
    print(result)
    del transformer_tokenizer

    # record special tokens map 'coz save_pretrained is not writing them
    with open(f"{tokenizer_dir_path}/special_tokens_map.json", mode='w+') as f:
        json.dump(
            {"unk_token": "[UNK]", "sep_token": "[SEP]", "pad_token": "[PAD]", "cls_token": "[CLS]", "mask_token": "[MASK]"},
            f,
            ensure_ascii=False
        )

    tokenizer_reloaded = AutoTokenizer.from_pretrained(str(tokenizer_dir_path))
    result2 = tokenizer_reloaded.tokenize('안녕 토크나이저!')
    print(result2)

    assert result == result2

    # same tokenizer.json is in $tokenizer_dir_path
    # os.remove(tokenizer_json_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="application of training vocabulary of BPE-tokenizers")

    parser.add_argument(
        "-t",
        "--tokenizer",
        help="the name of tokenizer which has been provided by huggingface",
        type=str,
        required=True,
        default='bert_word_piece'
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=False,
        default=None
    )

    parser.add_argument(
        '-cp',
        '--corpus_paths',
        help="corpus directory paths including corpus files",
        action='append',
        required=True
    )

    main(parser.parse_args())

