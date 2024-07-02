import json
import numpy as np
import os

from embeddings import GloveEmbedding, KazumaCharEmbedding
from pathlib import Path
from torch import nn as nn
from termcolor import colored
from tqdm import tqdm
from typing import List, Tuple, Dict, Union, Any

from tunip.constants import PAD_IDX
from tunip.logger import init_logging_handler_for_klass


class Embedding(nn.Embedding):
    def __init__(
        self, vocab, embedding_dim, word_embedding_filepath, padding_idx=PAD_IDX
    ):
        super(Embedding, self).__init__(
            vocab.size(), embedding_dim, padding_idx=PAD_IDX
        )

        self.logger = init_logging_handler_for_klass(klass=self.__class__)

        self.vocab = vocab
        self.word_embedding_filepath = word_embedding_filepath

    def load(self, filepath):
        self.logger.info(
            f"{filepath} loaded and its size of vocab is {self.vocab.size()}"
        )
        if os.path.exists(filepath):
            with open(filepath) as f:
                vecs = json.load(f)
                if len(vecs) != len(self.vocab):
                    # self.weight.data.size(0) == len(self.vocab)
                    self.logger.warn(f"the size of existing word embedding vector: {len(vecs)} and size of vocab: {len(self.vocab)} are different...")
                    os.remove(filepath)
        if not os.path.exists(filepath):
            self._dump_pretrained_emb(
                self.vocab.word2index,
                self.vocab.index2word,
                dump_path=filepath,
                embedding_dim=self.embedding_dim,
            )
            with open(filepath) as f:
                vecs = json.load(f)
        self.weight.data.copy_(self.weight.data.new(vecs))
        self.weight.requires_grad = True

    def fix(self):
        super().weight.requires_grad = False

    @staticmethod
    def _dump_pretrained_emb(word2index, index2word, dump_path, embedding_dim=100):
        print("Dumping pretrained embeddings...")
        embeddings = [
            GloveEmbedding(name="wikipedia_gigaword", d_emb=embedding_dim),
            # KazumaCharEmbedding(), # TODO compare to CharBiLSTM
        ]
        emb_list = []
        for i in tqdm(range(len(word2index.keys()))):
            w = index2word[i]
            e = []
            for emb in embeddings:
                e += emb.emb(w, default="zero")
            emb_list.append(e)
        Path(dump_path).parent.mkdir(parents=True, exist_ok=True)
        with open(dump_path, "wt") as f:
            json.dump(emb_list, f, ensure_ascii=False)
