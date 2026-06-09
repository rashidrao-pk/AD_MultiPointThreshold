from .single_point import fit_single_point
from .multi_point import fit_multi_point


def fit_threshold(scores, config):

    method = config.threshold.method

    if method == "single_point":

        return fit_single_point(
            scores,
            config.threshold.percentile
        )
    
    if method == "multi_point":
        return fit_multi_point(
            scores=scores,
            quantiles=config.scoring.quantiles,
            percentile=config.threshold.percentile,
            decision_rule=config.threshold.decision_rule,
        )
    

    raise ValueError(f"Unknown thresholding method {method}")