from .single_point import fit_single_point


def fit_threshold(scores, config):

    method = config.threshold.method

    if method == "single_point":

        return fit_single_point(
            scores,
            config.threshold.percentile
        )

    

    raise ValueError(f"Unknown thresholding method {method}")