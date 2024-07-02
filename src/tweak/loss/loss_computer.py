from abc import abstractmethod

from pytorch_metric_learning import miners, losses


class LossComputer:

    @abstractmethod
    def __call__(self, embeddings, labels):
        pass


class TripletMarginLossComputer:

    def __init__(self):
        self.miner = miners.MultiSimilarityMiner()
        self.loss_func = losses.TripletMarginLoss()

    def __call__(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)

        return loss
