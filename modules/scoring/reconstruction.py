import torch


def reconstruction_mse_score(loader, enc, dec, device):
    return _reconstruction_score(loader, enc, dec, device, metric="mse")


def reconstruction_l1_score(loader, enc, dec, device):
    return _reconstruction_score(loader, enc, dec, device, metric="l1")


def reconstruction_l2_score(loader, enc, dec, device):
    return _reconstruction_score(loader, enc, dec, device, metric="l2")


def _reconstruction_score(loader, enc, dec, device, metric="mse"):
    enc.eval()
    dec.eval()

    scores = []
    labels = []

    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device)

            mu, logvar = enc(images)
            recon = dec(mu)

            diff = images - recon

            if metric == "mse":
                batch_scores = (diff ** 2).mean(dim=(1, 2, 3))

            elif metric == "l1":
                batch_scores = diff.abs().mean(dim=(1, 2, 3))

            elif metric == "l2":
                batch_scores = torch.sqrt((diff ** 2).sum(dim=(1, 2, 3)))

            else:
                raise ValueError(f"Unknown reconstruction metric: {metric}")

            # CPU for thresholding and evalution later on...
            # to change  if necessary... 
            scores.append(batch_scores.detach().cpu())
            labels.append(batch_labels.detach().cpu())

    return torch.cat(scores), torch.cat(labels)


def reconstruction_quantile_score(loader, enc, dec, device, quantiles):
    enc.eval()
    dec.eval()

    scores = []
    labels = []

    q = torch.tensor(quantiles, device=device)

    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device)

            mu, logvar = enc(images)
            recon = dec(mu)

            anomaly_map = (images - recon).abs()

            flat_map = anomaly_map.view(anomaly_map.size(0), -1)

            batch_scores = torch.quantile(
                flat_map,
                q,
                dim=1,
            ).T

            scores.append(batch_scores.detach().cpu())
            labels.append(batch_labels.detach().cpu())

    return torch.cat(scores), torch.cat(labels)