import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from tunip.logger import init_logging_handler_for_klass


class BiLSTMEncoder(nn.Module):
    """
    BILSTM encoder.
    output the score of all labels.
    """

    def __init__(
        self,
        label_size: int,
        input_dim: int,
        hidden_dim: int,
        drop_lstm: float = 0.5,
        num_lstm_layers: int = 1,
    ):
        super(BiLSTMEncoder, self).__init__()

        self.logger = init_logging_handler_for_klass(klass=self.__class__)

        self.logger.info("Input size for Encoder: {}".format(input_dim))
        self.logger.info("Hidden size for Encoder: {}".format(hidden_dim))

        self.label_size = label_size
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim // 2,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.drop_lstm = nn.Dropout(drop_lstm)
        self.hidden2tag = nn.Linear(hidden_dim, self.label_size)

    def forward(
        self, word_rep: torch.Tensor, word_seq_lens: torch.Tensor
    ) -> torch.Tensor:
        """
        Encoding the input with BiLSTM
        :param word_rep: (batch_size, sent_len, input rep size)
        :param word_seq_lens: (batch_size, 1)
        :return: emission scores (batch_size, sent_len, hidden_dim)
        """
        sorted_seq_len, perm_idx = word_seq_lens.sort(0, descending=True)
        _, recover_idx = perm_idx.sort(0, descending=False)
        sorted_seq_tensor = word_rep[perm_idx]

        packed_words = pack_padded_sequence(sorted_seq_tensor, sorted_seq_len.cpu(), True)

        # packed_words [sum_of_lengths]x150 -> lstm_out [sum_of_lengths]x200
        lstm_out, _ = self.lstm(packed_words, None)
        # packed_lstm_out 6x[max_of_lengths]x200
        packed_lstm_out, _ = pad_packed_sequence(
            lstm_out, batch_first=True
        )  ## CARE: make sure here is batch_first, otherwise need to transpose.
        feature_out = self.drop_lstm(packed_lstm_out)

        outputs = self.hidden2tag(feature_out)
        return outputs[recover_idx]
