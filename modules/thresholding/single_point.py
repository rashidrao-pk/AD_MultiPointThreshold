import torch

from .base import ThresholdModel


class SinglePointThreshold(ThresholdModel):
    """Threshold model that compares scalar scores against one cutoff."""

    def __init__(self, threshold):
        """Store the scalar threshold value."""
        self.threshold = threshold

    def predict(self, scores):
        """Return 1 for scores above the threshold and 0 otherwise."""

        scores = scores.float().view(-1)

        return (scores > self.threshold).int()


def fit_single_point(scores, percentile=95):
    """Fit a scalar threshold from a percentile of training scores."""

    threshold = torch.quantile(
        scores.float().view(-1),
        percentile/100
    )

    return SinglePointThreshold(threshold)
