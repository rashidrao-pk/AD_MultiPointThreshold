import torch


def reconstruction_mse_score(loader, enc, dec, device):

    enc.eval()
    dec.eval()

    scores = []
    labels = []

    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device)

            mu, logvar = enc(images)
            recon = dec(mu)

            # one score per image
            batch_scores = ((images - recon) ** 2).mean(dim=(1, 2, 3))

            scores.append(batch_scores)
            labels.append(batch_labels.detach().cpu())

    scores = torch.cat(scores)
    labels = torch.cat(labels)

    return scores, labels