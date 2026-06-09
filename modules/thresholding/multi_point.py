import torch
from .base import ThresholdModel


class MultiPointThreshold(ThresholdModel):
    def __init__(self, quantiles, thresholds, decision_rule="any"):
        self.quantiles = quantiles
        self.thresholds = thresholds.float().view(-1)
        self.decision_rule = decision_rule

    def predict(self, scores):
        """
        scores shape:
            [N, K]

        where K = number of quantiles.
        and N is batch


        output shape: [N]
        """

        if not torch.is_tensor(scores):
            scores = torch.tensor(scores, dtype=torch.float32)

        scores = scores.float()

        exceed = scores > self.thresholds.view(1, -1)

        if self.decision_rule == "any":
            return exceed.any(dim=1).int()

        if self.decision_rule == "all":
            return exceed.all(dim=1).int()

        if self.decision_rule == "majority":
            # more than half 
            required = scores.shape[1] // 2 + 1
            return (exceed.sum(dim=1) >= required).int()

        raise ValueError(f"Unknown decision_rule: {self.decision_rule}")

    

def fit_multi_point(scores, quantiles, percentile=95, decision_rule="any"):
    """
    scores shape:
        [N, K]

    Example:
        N = number of train images
        K = number of quantiles
    """

    if not torch.is_tensor(scores):
        scores = torch.tensor(scores, dtype=torch.float32)

    scores = scores.float()

    if scores.dim() != 2:
        raise ValueError(
            f"Expected scores with shape [N, K], got {scores.shape}"
        )

    thresholds = torch.quantile(
        scores,
        percentile / 100.0,
        dim=0,
    )

    return MultiPointThreshold(
        quantiles=quantiles,
        thresholds=thresholds,
        decision_rule=decision_rule,
    )