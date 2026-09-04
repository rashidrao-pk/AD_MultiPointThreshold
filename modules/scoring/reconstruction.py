import torch
from tqdm import tqdm


def reconstruction_mse_score(loader, enc, dec, device):
    """Score samples by mean squared reconstruction error."""
    return _reconstruction_score(loader, enc, dec, device, metric="mse")


def reconstruction_l1_score(loader, enc, dec, device):
    """Score samples by mean absolute reconstruction error."""
    return _reconstruction_score(loader, enc, dec, device, metric="l1")


def reconstruction_l2_score(loader, enc, dec, device):
    """Score samples by L2 reconstruction error."""
    return _reconstruction_score(loader, enc, dec, device, metric="l2")


def _reconstruction_score(loader, enc, dec, device, metric="mse"):
    """Compute reconstruction scores and labels for one scalar error metric."""
    enc.eval()
    dec.eval()

    scores = []
    labels = []

    with torch.inference_mode():
        for data, batch_labels in tqdm(loader, desc=f"Scoring ({metric})"):
            data = data.to(device, non_blocking=True)

            mu, _ = enc(data)
            recon = dec(mu)

            diff = data - recon

            if metric == "mse":
                batch_scores = (diff**2).mean(dim=(1, 2, 3))
            elif metric == "l1":
                batch_scores = diff.abs().mean(dim=(1, 2, 3))
            elif metric == "l2":
                batch_scores = torch.sqrt((diff**2).sum(dim=(1, 2, 3)))
            else:
                raise ValueError(f"Unknown reconstruction metric: {metric}")

            scores.append(batch_scores.detach().cpu())
            labels.append(batch_labels.detach().cpu())

    return torch.cat(scores), torch.cat(labels)


def reconstruction_quantile_score(loader, enc, dec, device, quantiles):
    """Score samples using quantiles of per-pixel absolute reconstruction error."""
    enc.eval()
    dec.eval()

    scores = []
    labels = []

    q = torch.tensor(quantiles, device=device)

    with torch.inference_mode():
        for data, batch_labels in tqdm(loader, desc="Scoring (quantiles)"):
            data = data.to(device, non_blocking=True)
            # DEBUGGING
            # print(f"Data shape: {data.shape}")
            mu, _ = enc(data)
            recon = dec(mu)

            anomaly_map = (
                data - recon
            ).abs().mean(dim=1)  ## TODO: we can also try squared error here, but for now we stick to absolute error as it is more interpretable and less sensitive to outliers.
            flat_map = anomaly_map.flatten(start_dim=1)

            batch_scores = torch.quantile(
                flat_map,
                q,
                dim=1,
            ).T

            scores.append(batch_scores.detach().cpu())
            labels.append(batch_labels.detach().cpu())

    return torch.cat(scores), torch.cat(labels)
