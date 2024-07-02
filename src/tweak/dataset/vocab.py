import os
import pickle
import random

from pathlib import Path
from tunip.constants import PAD, BOS, EOS, UNK
from tunip.constants import PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX


class MemVocab:

    def __init__(self, ensure_special_tokens=True):
        self.word2index = {}
        if ensure_special_tokens:
            self.index2word = {PAD_IDX: PAD, BOS_IDX: BOS, EOS_IDX: EOS, UNK_IDX: UNK}
        else:
            self.index2word = {}
        self.n_words = len(self.index2word)  # Count default tokens
        self.word2index = dict([(v, k) for k, v in self.index2word.items()])

    def __len__(self):
        return self.n_words

    def size(self):
        return self.n_words

    def index_words(self, sent, input_type):
        if input_type == 'utter':
            for word in sent.split():
                self.index_word(word)
        elif input_type == 'slot':
            for slot in sent:
                d, s = slot.split("-")
                self.index_word(d)
                for ss in s.split():
                    self.index_word(ss)
        elif input_type == 'domain':
            for domain in sent:
                self.index_word(domain)
        elif input_type == 'word':
            for w in sent:
                self.index_word(w)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

    def get_vocabs(self, indices):
        return [self.index2word[idx] for idx in indices]


class Vocab:

    def __init__(self, dir_name, file_name, ensure_special_tokens=True):
        self.mem_vocab = MemVocab(ensure_special_tokens=ensure_special_tokens)

        # mkdir if vocab directory is not exists
        vocab_dir_path = Path(dir_name)
        if not vocab_dir_path.is_dir():
            vocab_dir_path.mkdir(parents=True, exist_ok=False)
        self.vocab_file_path = Path(f"{dir_name}/{file_name}")

        # load if the vocab file exists
        if self.vocab_file_path.exists():
            vocab_obj = self.load()
            self.copy(vocab_obj)

    def __len__(self):
        return self.mem_vocab.__len__()

    @property
    def word2index(self):
        return self.mem_vocab.word2index

    @property
    def index2word(self):
        return self.mem_vocab.index2word

    @property
    def n_words(self):
        return self.mem_vocab.n_words

    def size(self):
        return self.mem_vocab.__len__()

    def copy(self, other):
        self.mem_vocab.word2index = other.mem_vocab.word2index
        self.mem_vocab.index2word = other.mem_vocab.index2word
        self.mem_vocab.n_words = other.mem_vocab.n_words
        self.vocab_file_path = other.vocab_file_path

    def index_words(self, sent, input_type):
        self.mem_vocab.index_words(sent, input_type)

    def index_word(self, word):
        self.mem_vocab.index_word(word)

    def get_vocabs(self, indices):
        return self.mem_vocab.get_vocabs(indices)

    def save(self):
        if os.path.exists(self.vocab_file_path) is False:
            with open(self.vocab_file_path, 'wb') as f:
                pickle.dump(self, f)
        else:
            tmp_file_path = f'{self.vocab_file_path}.{random.randint(0, 1024*1024*1024)}'
            os.rename(self.vocab_file_path, tmp_file_path)
            with open(self.vocab_file_path, 'wb') as f:
                pickle.dump(self, f)
            os.remove(tmp_file_path)

    def load(self):
        obj = None
        if self.vocab_file_path.exists():
            with open(self.vocab_file_path, 'rb') as f:
                obj = pickle.load(f)
        return obj

    def index_merge_with(self, other):
        for e in other.word2index.items():
            # index the word of other vocab
            self.index_word(e[0])
        self.save()
    
    def remove_words(self, words):
        for w in words:
            index = self.mem_vocab.word2index[w]
            try:
                del self.mem_vocab.word2index[w]
            except KeyError:
                pass
            try:
                del self.mem_vocab.index2word[index]
            except KeyError:
                pass
