import argparse
import pickle
import time

from pathlib import Path
from typing import List

from transformers import AutoTokenizer

from tunip.corpus.corpus_path_provider import CorpusPathProviderFactory
from tunip.preprocess import preprocess_korean

from tweak.pretrain.document_example_builder import BertDocumentExampleBuilder
from tweak.pretrain.examples import PretrainingModelExample


class PretrainingDataPacker:

    def __init__(self, tokenizer, max_length):

        self.documents = None

        self.max_length = max_length
        self.max_prediction_per_sentence = 80

        self.tokenizer = tokenizer
        vocab_tokens = []
        for t in sorted([(v[1], v[0]) for v in tokenizer.get_vocab().items()], key=lambda v: v[0]):
            vocab_tokens.append(t[1])
        
        self.vocab_tokens = vocab_tokens
        self.examples = []
        self.length = -1


    def __len__(self):
        return self.length
    

    def __getstate__(self):
        state = self.__dict__.copy()
        return state
    

    def __setstate__(self, state):
        self.__dict__.update(state)
    

    def preprocess(self, text):
        text = pattern.sub(' ', text)
        text = url_pattern.sub('', text)
        text = text.strip()
        return


    def pack(self, filepath):
        """
        filepath: must be .txt file that includes doubled new-lines separated documents having new-line separated sentences
        """
        self.examples = self._build_examples(filepath, self.tokenizer)
        self.length = len(self.examples)


    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
    

    @staticmethod
    def load(filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)


    def _build_examples(self, path, tokenizer):
        examples = []
        documents = self._read_docs_from_file(path, tokenizer)

        document_example_builder = BertDocumentExampleBuilder(documents, len(documents), self.max_length, self.max_prediction_per_sentence, tokenizer, self.vocab_tokens)

        for document_id, document in enumerate(documents):
            examples_of_doc: List[PretrainingModelExample] = document_example_builder.build(document, document_id, False)
            # examples_of_doc: List[PretrainingModelExample] = document_example_builder.build(document, document_id, document_id % 1000 == 0)
            examples.extend(examples_of_doc)
            if document_id % 1000 == 0:
                print(f"build {len(examples)} examples; append {len(examples_of_doc)} examples from document:[{document_id}] ...")

        return examples


    def _read_docs_from_file(self, path, tokenizer):
        print(f"START TO READ DOCS FROM DOC-LINE-FILE: {path}")
        count = 0

        documents = []
        document = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                # print(line)
                if line == "\n":
                    if document:
                        documents.append(document)
                    document = []
                    count += 1
                    if count % 100 == 0:
                        print(f"# of reading docs: {count}")
                else:
                    # filter korean strictly
                    preprocessed = preprocess_korean(line, strict=True)
                    if preprocessed:
                        document.append(tokenizer.tokenize(preprocessed))
                
        print(f"END TO READ DOCS FROM DOC-LINE-FILE: {path}")
        return documents


def run(args):
    tokenizer_name_or_path = args.tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)
    corpus_bin_path = Path(args.corpus_bin_path)
    corpus_bin_path.mkdir(parents=True, exist_ok=True)

    path_provider = CorpusPathProviderFactory.create(
        corpus_type="dir",
        input_dir=args.corpus_shd_path,
        extensions=[".txt"],
        split_path=""
    )

    t = time.process_time()

    for filepath in path_provider():
        print(filepath)
        bin_filename = Path(filepath).with_suffix(".bin").name

        # pack and save pretraining data
        pretrain_data_packer = PretrainingDataPacker(tokenizer=tokenizer, max_length=args.max_length)
        pretrain_data_packer.pack(filepath)
        pretrain_data_packer.save(f"{corpus_bin_path}/{bin_filename}")
    
    print(f"It takes about {(time.process_time()-t)/60.} minutes.")
    print(f"END TO pack pre-training data from {args.corpus_shd_path} to {args.corpus_bin_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pre-training data packer")

    parser.add_argument(
        "-t",
        "--tokenizer",
        help="the name or path of tokenizer which has been provided by huggingface",
        type=str,
        required=True
    )

    parser.add_argument(
        "-cptp",
        "--corpus_shd_path",
        type=str,
        required=True
    )

    parser.add_argument(
        "-cpbp",
        "--corpus_bin_path",
        type=str,
        required=True
    )

    parser.add_argument(
        "-len",
        "--max_length",
        type=int,
        required=True
    )
    
    run(args=parser.parse_args())
