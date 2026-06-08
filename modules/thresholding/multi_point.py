from .base import ThresholdModel


class MultiPointThreshold(ThresholdModel):

    def __init__(self, quantiles):
        self.quantiles = quantiles

    def predict(self, scores):

        # TODO
        # calcualte the quantiles on the anomaly map
        raise NotImplementedError


def fit_multi_point(features, config):

    return MultiPointThreshold(
        config.threshold.quantiles
    )