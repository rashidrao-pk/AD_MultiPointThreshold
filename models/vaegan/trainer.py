from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch import optim
from tqdm import tqdm

from .checkpoint import build_dataset_summary, save_checkpoint
from .model import Decoder, Discriminator, Encoder


def _cfg_get(obj, name, default=None):
    return getattr(obj, name, default) if obj is not None else default


def _reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    return mu + torch.randn_like(std) * std


def _to_image(tensor):
    return ((tensor.detach().cpu().clamp(-1, 1) + 1) / 2).permute(1, 2, 0).numpy()


def _plot_history(loss_history, output_path):
    if not loss_history:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history = pd.DataFrame(loss_history)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for ax, column, title in (
        (axes[0, 0], "recon_loss", "Reconstruction loss"),
        (axes[0, 1], "vae_loss", "VAE-GAN generator loss"),
        (axes[1, 0], "disc_loss", "Discriminator loss"),
        (axes[1, 1], "val_recon_loss", "Validation reconstruction loss"),
    ):
        if column in history:
            ax.plot(history.index + 1, history[column], linewidth=1.3)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.25)
        else:
            ax.set_axis_off()

    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_reconstruction_preview(encoder, decoder, loader, device, output_path, max_images=6):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    encoder.eval()
    decoder.eval()
    images, _ = next(iter(loader))
    images = images[:max_images].to(device)

    with torch.inference_mode():
        mu, _ = encoder(images)
        recon = decoder(mu)

    fig, axes = plt.subplots(2, len(images), figsize=(2.4 * len(images), 4.8), constrained_layout=True)
    if len(images) == 1:
        axes = axes.reshape(2, 1)
    for idx in range(len(images)):
        axes[0, idx].imshow(_to_image(images[idx]))
        axes[0, idx].set_title("input", fontsize=8)
        axes[0, idx].axis("off")
        axes[1, idx].imshow(_to_image(recon[idx]))
        axes[1, idx].set_title("recon", fontsize=8)
        axes[1, idx].axis("off")

    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def train_one_epoch(
    encoder,
    decoder,
    discriminator,
    train_loader,
    optimizer_enc_dec,
    optimizer_dis,
    device,
    beta_kl=1e-4,
    beta_gan=1e-4,
):
    encoder.train()
    decoder.train()
    discriminator.train()

    reconstruction_loss_fn = nn.MSELoss()
    adversarial_loss_fn = nn.BCEWithLogitsLoss()
    totals = {
        "recon_loss": 0.0,
        "kl_loss": 0.0,
        "gan_loss": 0.0,
        "beta_kl_loss": 0.0,
        "beta_gan_loss": 0.0,
        "vae_loss": 0.0,
        "disc_loss": 0.0,
    }
    dis_preds = []
    dis_labels = []

    for images, _ in tqdm(train_loader, desc="train batches", leave=False):
        images = images.to(device, non_blocking=True)
        batch_size = images.size(0)

        optimizer_enc_dec.zero_grad()
        mu, logvar = encoder(images)
        z = _reparameterize(mu, logvar)
        recon = decoder(z)

        recon_loss = reconstruction_loss_fn(recon, images)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        fake_logits_for_gen = discriminator(recon)
        gan_loss = adversarial_loss_fn(fake_logits_for_gen, torch.ones_like(fake_logits_for_gen))
        vae_loss = recon_loss + beta_kl * kl_loss + beta_gan * gan_loss
        vae_loss.backward()
        optimizer_enc_dec.step()

        optimizer_dis.zero_grad()
        real_logits = discriminator(images)
        fake_logits = discriminator(recon.detach())
        real_loss = adversarial_loss_fn(real_logits, torch.ones_like(real_logits))
        fake_loss = adversarial_loss_fn(fake_logits, torch.zeros_like(fake_logits))
        disc_loss = (real_loss + fake_loss) / 2
        disc_loss.backward()
        optimizer_dis.step()

        real_preds = (torch.sigmoid(real_logits).detach().cpu().numpy() > 0.5).astype(int)
        fake_preds = (torch.sigmoid(fake_logits).detach().cpu().numpy() > 0.5).astype(int)
        dis_preds.extend(real_preds.flatten())
        dis_labels.extend([1] * batch_size)
        dis_preds.extend(fake_preds.flatten())
        dis_labels.extend([0] * batch_size)

        totals["recon_loss"] += recon_loss.item()
        totals["kl_loss"] += kl_loss.item()
        totals["gan_loss"] += gan_loss.item()
        totals["beta_kl_loss"] += beta_kl * kl_loss.item()
        totals["beta_gan_loss"] += beta_gan * gan_loss.item()
        totals["vae_loss"] += vae_loss.item()
        totals["disc_loss"] += disc_loss.item()

    n_batches = max(len(train_loader), 1)
    metrics = {key: value / n_batches for key, value in totals.items()}
    metrics["dis_acc"] = accuracy_score(dis_labels, dis_preds) if dis_labels else 0.0
    metrics["dis_f1"] = f1_score(dis_labels, dis_preds) if dis_labels else 0.0
    return metrics


def validate(encoder, decoder, val_loader, device):
    encoder.eval()
    decoder.eval()
    reconstruction_loss_fn = nn.MSELoss()
    total = 0.0
    n_batches = 0

    with torch.inference_mode():
        for images, _ in tqdm(val_loader, desc="val batches", leave=False):
            images = images.to(device, non_blocking=True)
            mu, _ = encoder(images)
            recon = decoder(mu)
            total += reconstruction_loss_fn(recon, images).item()
            n_batches += 1

    return {"val_recon_loss": total / max(n_batches, 1)}


def train_model(config, train_loader, val_loader, train_dataset, val_dataset, run_dir, device):
    training_cfg = _cfg_get(config, "training", None)
    model_cfg = config.model
    model_name = str(_cfg_get(model_cfg, "name", "simple_vaegan"))
    is_advis = model_name == "advis_vaegan"
    model_label = "ADVIS-compatible VAE-GAN" if is_advis else "Simple VAE-GAN"

    epochs = int(_cfg_get(training_cfg, "epochs", 20))
    lr_enc_dec = float(_cfg_get(training_cfg, "learning_rate_enc_dec", 1e-3))
    lr_dis = float(_cfg_get(training_cfg, "learning_rate_dis", 1e-4))
    beta_kl = float(_cfg_get(training_cfg, "beta_kl", 1e-4))
    beta_gan = float(_cfg_get(training_cfg, "beta_gan", 1e-4))
    save_every = int(_cfg_get(training_cfg, "save_every", 1))
    latent_dim = int(_cfg_get(model_cfg, "latent_dim", 64))

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    encoder = Encoder(z_size=latent_dim).to(device)
    decoder = Decoder(z_size=latent_dim).to(device)
    discriminator = Discriminator().to(device)
    optimizer_enc_dec = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr_enc_dec)
    optimizer_dis = optim.Adam(discriminator.parameters(), lr=lr_dis)

    dataset_summary = build_dataset_summary(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
    )
    loss_history = []
    best_val_loss = float("inf")

    for epoch in tqdm(range(1, epochs + 1), desc="epochs"):
        train_metrics = train_one_epoch(
            encoder,
            decoder,
            discriminator,
            train_loader,
            optimizer_enc_dec,
            optimizer_dis,
            device,
            beta_kl=beta_kl,
            beta_gan=beta_gan,
        )
        val_metrics = validate(encoder, decoder, val_loader, device)
        epoch_metrics = {"epoch": epoch, **train_metrics, **val_metrics}
        loss_history.append(epoch_metrics)

        pd.DataFrame(loss_history).to_csv(run_dir / "loss_history.csv", index=False)
        _plot_history(loss_history, run_dir / "training_curves.png")
        _save_reconstruction_preview(encoder, decoder, val_loader, device, run_dir / "reconstruction_preview.png")

        if epoch % save_every == 0 or epoch == epochs:
            save_checkpoint(
                run_dir / "model_last.pt",
                encoder,
                decoder,
                discriminator,
                optimizer_enc_dec=optimizer_enc_dec,
                optimizer_dis=optimizer_dis,
                epoch=epoch,
                loss_history=loss_history,
                config=config,
                dataset_summary=dataset_summary,
                metrics=epoch_metrics,
                model_name=model_name,
                notes=f"Latest {model_label} checkpoint for this training run.",
            )

        if val_metrics["val_recon_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_recon_loss"]
            save_checkpoint(
                run_dir / "model_best.pt",
                encoder,
                decoder,
                discriminator,
                optimizer_enc_dec=optimizer_enc_dec,
                optimizer_dis=optimizer_dis,
                epoch=epoch,
                loss_history=loss_history,
                config=config,
                dataset_summary=dataset_summary,
                metrics=epoch_metrics,
                model_name=model_name,
                notes=f"Best {model_label} checkpoint by validation reconstruction loss.",
            )

        print(
            f"epoch={epoch:04d} "
            f"recon={epoch_metrics['recon_loss']:.6f} "
            f"val_recon={epoch_metrics['val_recon_loss']:.6f} "
            f"disc={epoch_metrics['disc_loss']:.6f}"
        )

    return {
        "encoder": encoder,
        "decoder": decoder,
        "discriminator": discriminator,
        "loss_history": loss_history,
        "best_val_loss": best_val_loss,
        "run_dir": run_dir,
    }
