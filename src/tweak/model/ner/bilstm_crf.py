import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from typing import Tuple

from tunip.logger import init_logging_handler_for_klass

from tweak.model.ner.bilstm.bilstm_encoder import BiLSTMEncoder
from tweak.model.ner.config.context_embedding_type import ContextualEmbeddingType
from tweak.model.ner.crf.linear_crf_inferencer import LinearCRF
from tweak.model.ner.embed.word_embedder import WordEmbedder
from tweak.utils.embed_utils import Embedding


class BiLstmCRF(nn.Module):

    def __init__(self, config):
        super(BiLstmCRF, self).__init__()

        self.logger = init_logging_handler_for_klass(klass=self.__class__)

        # TODO move if-else-blocks to the constructor of Embedding
        if config.word_embedding_filepath is None:
            self.logger.info(f'No word embedding file.')
            vocab = config.word_vocab
            scale = np.sqrt(3.0 / config.embedding_dim)
            word_embed = np.empty([vocab.size(), config.embedding_dim])
            for w in vocab.word2index:
                word_embed[vocab.word2index[w], :] = np.random.uniform(-scale, scale, [1, config.embedding_dim])
            word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(word_embed), freeze=False)
        else:
            word_embedding = Embedding(
                vocab=config.word_vocab, embedding_dim=config.embedding_dim, word_embedding_filepath=config.word_embedding_filepath
            )
            self.logger.info(f'word embedding filepath = {config.word_embedding_filepath}')
            word_embedding.load(Path(config.word_embedding_filepath).expanduser().absolute())

        self.embedder = WordEmbedder(
            word_embedding=word_embedding,
            embedding_dim=config.embedding_dim,
            context_emb_size=config.context_emb_size,
            use_char_rnn=config.use_char_rnn,
            char_emb_size=config.char_emb_size,
            char_size=len(config.char_vocab.word2index),
            char_hidden_size=config.charlstm_hidden_dim,
            # TODO static_context_emb=config.static_context_emb,
            static_context_emb=ContextualEmbeddingType.none,
            dropout=config.dropout,
        )
        self.encoder = BiLSTMEncoder(
            label_size=config.label_vocab.size(),
            input_dim=self.embedder.get_output_dim(),
            hidden_dim=config.hidden_dim,
            drop_lstm=config.dropout,
        )
        self.inferencer = LinearCRF(
            label_vocab=config.label_vocab
        )

        # TODO Without CRF option
        # self.entropy = nn.CrossEntropyLoss()

    def forward(
        self,
        words: torch.Tensor,
        word_seq_lens: torch.Tensor,
        context_emb: torch.Tensor,
        chars: torch.Tensor,
        char_seq_lens: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the negative loglikelihood.
        :param words: (batch_size x max_seq_len)
        :param word_seq_lens: (batch_size)
        :param context_emb: (batch_size x max_seq_len x context_emb_size)
        :param chars: (batch_size x max_seq_len x max_char_len)
        :param char_seq_lens: (batch_size x max_seq_len)
        :param labels: (batch_size x max_seq_len)
        :return: the total negative log-likelihood loss
        """
        word_rep = self.embedder(
            words, word_seq_lens, context_emb, chars, char_seq_lens
        )
        dev_num = word_rep.get_device()
        curr_dev = (
            torch.device(f"cuda:{dev_num}") if dev_num >= 0 else torch.device("cpu")
        )

        # TODO return encoder output in the forward func.
        lstm_scores = self.encoder(word_rep, word_seq_lens)

        # TODO params of the forward func. would be data_item
        # TODO mask preprocessing must be in to the dataset or data collocator
        batch_size = words.size(0)
        sent_len = words.size(1)
        mask_temp = (
            torch.arange(1, sent_len + 1, dtype=torch.long, device=curr_dev)
            .unsqueeze(axis=0)
            .expand(batch_size, sent_len)
        )
        mask = torch.le(
            mask_temp, word_seq_lens
            .unsqueeze(axis=1)
            .expand(batch_size, sent_len)
        )
        unlabed_score, labeled_score = self.inferencer(
            lstm_scores, word_seq_lens, labels, mask
        )
        return unlabed_score - labeled_score

        # TODO Without CRF option
        # loss = self.entropy(lstm_scores.transpose(1, 2), labels)
        # return loss

    def decode(
        self,
        words: torch.Tensor,
        word_seq_lens: torch.Tensor,
        context_emb: torch.Tensor,
        chars: torch.Tensor,
        char_seq_lens: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode the batch input
        :param batchInput:
        :return:
        """
        # TODO call self.forward()
        word_rep = self.embedder(
            words, word_seq_lens, context_emb, chars, char_seq_lens
        )
        # features = self.encoder(word_rep, word_seq_lens)
        features = self.encoder(word_rep, word_seq_lens.cpu())

        # return None, torch.argmax(features, dim=2)

        # batch
        # batch X seq
        best_scores, decode_idx = self.inferencer.decode(features, word_seq_lens)
        return best_scores, decode_idx
