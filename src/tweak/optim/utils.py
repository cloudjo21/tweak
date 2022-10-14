import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

from termcolor import colored
from transformers import AdamW
from typing import List, Tuple, Dict, Any

from tunip.logger import init_logging_handler


logger = init_logging_handler(name=os.path.basename(__file__))


def lr_decay(
    learning_rate: float, lr_decay: float, optimizer: optim.Optimizer, epoch: int
) -> optim.Optimizer:
    """
    Method to decay the learning rate
    :param config: configuration
    :param optimizer: optimizer
    :param epoch: epoch number
    :return:
    """
    lr = learning_rate / (1 + lr_decay * (epoch - 1))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    logger.info("learning rate is set to: {}".format(lr))

    return optimizer


def get_optimizer(
    optimizer: str,
    learning_rate: float,
    model: nn.Module,
    l2: float = 1e-8,
    eps: float = 1e-8,
    warmup_step: int = 0,
):
    if optimizer.lower() == "sgd":
        logger.info(
            colored(
                "Using SGD: lr is: {}, L2 regularization is: {}".format(
                    learning_rate, l2
                ),
                "yellow",
            )
        )
        return optim.SGD(model.parameters(), lr=learning_rate, weight_decay=float(l2))
    elif optimizer.lower() == "adam":
        logger.info(
            colored(f"Using Adam, with learning rate: {learning_rate}", "yellow")
        )
        return optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer.lower() == "adamw":
        logger.info(
            colored(
                f"Using AdamW optimizeer with {learning_rate} learning rate, "
                f"eps: {1e-8}",
                "yellow",
            )
        )
        return AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    else:
        logger.error("Illegal optimizer: {}".format(optimizer))
        exit(1)
