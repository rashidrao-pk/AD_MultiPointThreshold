from .reconstruction import reconstruction_mse_score



def score_samples(loader, enc, dec, disc, config):

    method = config.scoring.method

    if method == "reconstruction_mse":
        return reconstruction_mse_score(loader, enc, dec, config.device)

    
    else:
        raise ValueError(f"Unknown scoring method {method}")