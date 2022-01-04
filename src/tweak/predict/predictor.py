import torch


class Predictor(torch.nn.Module):

    def __init__(self, config):
        self.config = config 
