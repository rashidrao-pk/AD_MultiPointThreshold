from .reconstruction import (
    reconstruction_mse_score,
    reconstruction_l1_score,
    reconstruction_l2_score,
    reconstruction_quantile_score,
)


def score_samples(loader, enc, dec, disc, config):
    """Dispatch sample scoring to the method selected in the config."""

    method = config.scoring.method
    device = config.device

    if method == "reconstruction_mse":
        return reconstruction_mse_score(loader, enc, dec, device)

    if method == "reconstruction_l1":
        return reconstruction_l1_score(loader, enc, dec, device)

    if method == "reconstruction_l2":
        return reconstruction_l2_score(loader, enc, dec, device)

    if method == "reconstruction_quantiles":
        return reconstruction_quantile_score(
            loader,
            enc,
            dec,
            device,
            config.scoring.quantiles,
        )

    raise ValueError(f"Unknown scoring method: {method}")
