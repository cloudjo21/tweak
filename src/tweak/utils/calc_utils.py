import torch


def log_sum_exp(vec: torch.Tensor) -> torch.Tensor:
    """
    Calculate the log_sum_exp trick for the tensor.
    :param vec: [batch_size * from_label * to_label].
    :return: [batch_size * to_label]
    """
    max_scores, idx = torch.max(vec, 1)
    max_scores[max_scores == -float("Inf")] = 0
    # max_scores_expanded = max_scores.view(vec.shape[0], 1, vec.shape[2]).expand(
    max_scores_expanded = max_scores.reshape(vec.shape[0], 1, vec.shape[2]).expand(
        vec.shape[0], vec.shape[1], vec.shape[2]
    )
    return max_scores + torch.log(torch.sum(torch.exp(vec - max_scores_expanded), 1))
