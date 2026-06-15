import torch

from .base import ThresholdModel


class SinglePointThreshold(ThresholdModel):

    def __init__(self, threshold):
        self.threshold = threshold

    def predict(self, scores):

        scores = scores.float().view(-1)

        return (scores > self.threshold).int()


def fit_single_point(scores, percentile=95):

    threshold = torch.quantile(
        scores.float().view(-1),
        percentile/100
    )
    

    return SinglePointThreshold(threshold)