import torch.nn as nn
import torch

from typing import Dict, List, Tuple

from tunip.constants import BOS, EOS, PAD

from tweak.dataset.vocab import Vocab
from tweak.utils.calc_utils import log_sum_exp


class LinearCRF(nn.Module):
    def __init__(
        self,
        label_vocab: Vocab,
        add_iobes_constraint: bool = False,
    ):
        super(LinearCRF, self).__init__()

        self.label_size = label_vocab.size()
        self.label2idx = label_vocab.word2index
        self.idx2labels = label_vocab.index2word

        self.start_idx = self.label2idx[BOS]
        self.end_idx = self.label2idx[EOS]
        self.pad_idx = self.label2idx[PAD]

        # initialize the following transition (anything never cannot -> start. end never  cannot-> anything. Same thing for the padding label)
        init_transition = torch.randn(self.label_size, self.label_size)
        init_transition[:, self.start_idx] = -10000.0
        init_transition[self.end_idx, :] = -10000.0
        init_transition[:, self.pad_idx] = -10000.0
        init_transition[self.pad_idx, :] = -10000.0

        self.transition = nn.Parameter(init_transition)


    def forward(self, lstm_scores, word_seq_lens, tags, mask):
        """
        Calculate the negative log-likelihood
        :param lstm_scores:
        :param word_seq_lens:
        :param tags:
        :param mask:
        :return:
        """
        all_scores = self.calculate_all_scores(lstm_scores=lstm_scores)
        unlabed_score = self.forward_unlabeled(all_scores, word_seq_lens)
        labeled_score = self.forward_labeled(all_scores, word_seq_lens, tags, mask)
        return unlabed_score, labeled_score


    def forward_unlabeled(
        self, all_scores: torch.Tensor, word_seq_lens: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the scores with the forward algorithm. Basically calculating the normalization term
        :param all_scores: (batch_size x max_seq_len x num_labels x num_labels) from (lstm scores + transition scores).
        :param word_seq_lens: (batch_size)
        :return: The score for all the possible structures.
        """
        batch_size = all_scores.size(0)
        seq_len = all_scores.size(1)
        dev_num = all_scores.get_device()
        curr_dev = (
            torch.device(f"cuda:{dev_num}") if dev_num >= 0 else torch.device("cpu")
        )
        alpha = torch.zeros(batch_size, seq_len, self.label_size, device=curr_dev)

        alpha[:, 0, :] = all_scores[
            :, 0, self.start_idx, :
        ]  ## the first position of all labels = (the transition from start - > all labels) + current emission.

        for word_idx in range(1, seq_len):
            ## batch_size, self.label_size, self.label_size
            before_log_sum_exp = (
                alpha[:, word_idx - 1, :]
                .reshape(batch_size, self.label_size, 1)
                .expand(batch_size, self.label_size, self.label_size)
                + all_scores[:, word_idx, :, :]
            )
            alpha[:, word_idx, :] = log_sum_exp(before_log_sum_exp)

        ### batch_size x label_size
        last_alpha = torch.gather(
            alpha,
            1,
            word_seq_lens
            .reshape(batch_size, 1, 1)
            .expand(batch_size, 1, self.label_size)
            - 1,
        ).reshape(batch_size, self.label_size)
        last_alpha += (
            self.transition[:, self.end_idx]
            .reshape(1, self.label_size)
            .expand(batch_size, self.label_size)
        )
        last_alpha = log_sum_exp(
                last_alpha.unsqueeze(axis=-1)
            ).reshape(batch_size)

        ## final score for the unlabeled network in this batch, with size: 1
        return torch.sum(last_alpha)


    def forward_labeled(
        self,
        all_scores: torch.Tensor,
        word_seq_lens: torch.Tensor,
        tags: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the scores for the gold instances.
        :param all_scores: (batch, seq_len, label_size, label_size)
        :param word_seq_lens: (batch, seq_len)
        :param tags: (batch, seq_len)
        :param masks: batch, seq_len
        :return: sum of score for the gold sequences Shape: (batch_size)
        """
        batch_size = all_scores.shape[0]
        sent_len = all_scores.shape[1]

        ## all the scores to current labels: batch, seq_len, all_from_label?
        current_tag_scores = torch.gather(
            all_scores,
            3,
            tags.unsqueeze(axis=-1).unsqueeze(axis=-1).expand(
                batch_size, sent_len, self.label_size, 1
            )
        ).reshape(batch_size, -1, self.label_size)
        if sent_len != 1:
            inter_tag_trans_scores = torch.gather(
                current_tag_scores[:, 1:, :],
                2,
                tags[:, : sent_len - 1].reshape(batch_size, sent_len - 1, 1)
            ).reshape(batch_size, -1)
        start_tag_trans_scores = current_tag_scores[:, 0, self.start_idx]
        end_tag_ids = torch.gather(tags, 1, word_seq_lens.reshape(batch_size, 1) - 1)
        end_tag_trans_scores = torch.gather(
            self.transition[:, self.end_idx]
            .reshape(1, self.label_size)
            .expand(batch_size, self.label_size),
            1,
            end_tag_ids,
        ).reshape(batch_size)
        score = torch.sum(start_tag_trans_scores) + torch.sum(end_tag_trans_scores)
        if sent_len != 1:
            score += torch.sum(inter_tag_trans_scores.masked_select(masks[:, 1:]))
        return score

    def calculate_all_scores(self, lstm_scores: torch.Tensor) -> torch.Tensor:
        """
        Calculate all scores by adding up the transition scores and emissions (from lstm).
        Basically, compute the scores for each edges between labels at adjacent positions.
        This score is later be used for forward-backward inference
        :param lstm_scores: emission scores.
        :return:
        """
        batch_size = lstm_scores.size(0)
        seq_len = lstm_scores.size(1)
        scores = self.transition.reshape(1, 1, self.label_size, self.label_size).expand(
            batch_size, seq_len, self.label_size, self.label_size
        ) + lstm_scores.reshape(batch_size, seq_len, 1, self.label_size).expand(
            batch_size, seq_len, self.label_size, self.label_size
        )
        return scores

    def decode(self, features, word_seq_lens) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode the batch input
        :param batchInput:
        :return:
        """
        all_scores = self.calculate_all_scores(features)
        best_scores, decode_idx = self.viterbi_decode(all_scores, word_seq_lens)
        return best_scores, decode_idx

    def viterbi_decode(
        self, all_scores: torch.Tensor, word_seq_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Use viterbi to decode the instances given the scores and transition parameters
        :param all_scores: (batch_size x max_seq_len x num_labels)
        :param word_seq_lens: (batch_size)
        :return: the best scores as well as the predicted label ids.
               (batch_size) and (batch_size x max_seq_len)
        """
        batch_size = all_scores.shape[0]
        sent_len = all_scores.shape[1]
        dev_num = all_scores.get_device()
        curr_dev = (
            torch.device(f"cuda:{dev_num}") if dev_num >= 0 else torch.device("cpu")
        )
        scores_record = torch.zeros(
            [batch_size, sent_len, self.label_size], device=curr_dev
        )
        idx_record = torch.zeros(
            [batch_size, sent_len, self.label_size], dtype=torch.int64, device=curr_dev
        )
        mask = torch.ones_like(word_seq_lens, dtype=torch.int64, device=curr_dev)

        start_ids = torch.full(
            (batch_size, self.label_size),
            self.start_idx,
            dtype=torch.int64,
            device=curr_dev,
        )
        decode_idx = torch.LongTensor(batch_size, sent_len).to(curr_dev)

        scores = all_scores
        scores_record[:, 0, :] = scores[
            :, 0, self.start_idx, :
        ]  ## represent the best current score from the start, is the best
        idx_record[:, 0, :] = start_ids
        for word_idx in range(1, sent_len):
            ### scores_idx: batch x from_label x to_label at current index.
            scores_idx = (
                scores_record[:, word_idx - 1, :]
                .view(batch_size, self.label_size, 1)
                .expand(batch_size, self.label_size, self.label_size)
                + scores[:, word_idx, :, :]
            )
            idx_record[:, word_idx, :] = torch.argmax(
                scores_idx, 1
            )  ## the best previous label idx to crrent labels
            scores_record[:, word_idx, :] = torch.gather(
                scores_idx,
                1,
                idx_record[:, word_idx, :].view(batch_size, 1, self.label_size),
            ).view(batch_size, self.label_size)

        last_scores = torch.gather(
            scores_record,
            1,
            word_seq_lens.view(batch_size, 1, 1).expand(batch_size, 1, self.label_size)
            - 1,
        ).view(
            batch_size, self.label_size
        )  ##select position
        last_scores += (
            self.transition[:, self.end_idx]
            .view(1, self.label_size)
            .expand(batch_size, self.label_size)
        )
        decode_idx[:, 0] = torch.argmax(last_scores, 1)
        best_scores = torch.gather(last_scores, 1, decode_idx[:, 0].view(batch_size, 1))

        for distance_to_last in range(sent_len - 1):
            last_idx_record = torch.gather(
                idx_record,
                1,
                torch.where(
                    word_seq_lens - distance_to_last - 1 > 0,
                    word_seq_lens - distance_to_last - 1,
                    mask,
                )
                .view(batch_size, 1, 1)
                .expand(batch_size, 1, self.label_size),
            ).view(batch_size, self.label_size)
            decode_idx[:, distance_to_last + 1] = torch.gather(
                last_idx_record, 1, decode_idx[:, distance_to_last].view(batch_size, 1)
            ).view(batch_size)

        return best_scores, decode_idx
