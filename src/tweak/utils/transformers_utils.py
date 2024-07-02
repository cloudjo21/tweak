import torch

from itertools import repeat


def length_of_input_ids(batch_input_ids: torch.Tensor):
    lengths = []
    for input_ids in batch_input_ids:
        try:
            lengths.append((input_ids == 0).nonzero()[0].item())
        except IndexError as ie:
            #IndexError: index 0 is out of bounds for dimension 0 with size 0
            lengths.append(len(input_ids))
    return lengths


def zero_indexes_of_input_ids(batch_input_ids: torch.Tensor):
    """
    return zero indexes for each input having own length individually
    """
    lengths = length_of_input_ids(batch_input_ids)

    zero_indexes = []
    for i, length in enumerate(lengths):
        zero_indexes.append(list(range(length, batch_input_ids[i].shape[0])))

    return zero_indexes


def index_fill_with_zero_vector(batch_input_ids: torch.Tensor, tensor: torch.Tensor, device: torch.device):
    """
    tensor: [batch]x[length]x[hidden]
    """
    # assert batch size between encoding and tensor
    assert len(batch_input_ids) == len(tensor)

    zero_indexes_list: list = zero_indexes_of_input_ids(batch_input_ids)

    zero_indexes_batch, zero_indexes_seq = [], []
    for bi, zero_indexes in enumerate(zero_indexes_list):
        zero_indexes_batch.extend(list(repeat(bi, len(zero_indexes))))
        zero_indexes_seq.extend(zero_indexes)
    
    # do index_put tensor with transposed zero_indexes and zero vector
    tensor_with_zeros = tensor.index_put(
        [
            torch.tensor(zero_indexes_batch, dtype=torch.int64, device=device),
            torch.tensor(zero_indexes_seq, dtype=torch.int64, device=device)
        ],
        torch.zeros(tensor.shape[-1], device=device),
        accumulate=False
    )
    return tensor_with_zeros
