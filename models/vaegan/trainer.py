from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch import optim
from tqdm import tqdm

from .checkpoint import build_dataset_summary, load_checkpoint, save_checkpoint
from .model import Decoder, Discriminator, Encoder


def _cfg_get(obj, name, default=None):
    """Read an optional config attribute with a default fallback."""
    return getattr(obj, name, default) if obj is not None else default


def _reparameterize(mu, logvar):
    """Sample latent vectors using the VAE reparameterization trick."""
    std = torch.exp(0.5 * logvar)
    return mu + torch.randn_like(std) * std


def _latent_distance_sq(mu, center):
    """Return squared latent distance from each sample to the center."""
    return torch.sum((mu - center.view(1, -1)) ** 2, dim=1)


def _svdd_loss(distances_sq, radius=0.0, nu=0.1):
    """Compute compact or soft-boundary SVDD loss from squared distances."""
    radius = float(radius)
    nu = max(float(nu), 1e-6)
    if radius <= 0.0:
        return torch.mean(distances_sq)

    radius_sq = radius**2
    return radius_sq + torch.mean(torch.relu(distances_sq - radius_sq)) / nu


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
    beta_center=0.0,
    beta_svdd=0.0,
    latent_center=None,
    svdd_radius=0.0,
    svdd_nu=0.1,
):
    """Train encoder, decoder, and discriminator for one epoch."""
    encoder.train()
    decoder.train()
    discriminator.train()

    reconstruction_loss_fn = nn.MSELoss()
    adversarial_loss_fn = nn.BCEWithLogitsLoss()
    totals = {
        "recon_loss": 0.0,
        "kl_loss": 0.0,
        "gan_loss": 0.0,
        "center_loss": 0.0,
        "svdd_loss": 0.0,
        "beta_kl_loss": 0.0,
        "beta_gan_loss": 0.0,
        "beta_center_loss": 0.0,
        "beta_svdd_loss": 0.0,
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

        center_loss = torch.zeros((), device=device)
        svdd_loss = torch.zeros((), device=device)
        if latent_center is not None and (beta_center > 0.0 or beta_svdd > 0.0):
            distances_sq = _latent_distance_sq(mu, latent_center)
            center_loss = torch.mean(distances_sq)
            svdd_loss = _svdd_loss(distances_sq, radius=svdd_radius, nu=svdd_nu)

        vae_loss = (
            recon_loss
            + beta_kl * kl_loss
            + beta_gan * gan_loss
            + beta_center * center_loss
            + beta_svdd * svdd_loss
        )
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
        totals["center_loss"] += center_loss.item()
        totals["svdd_loss"] += svdd_loss.item()
        totals["beta_kl_loss"] += beta_kl * kl_loss.item()
        totals["beta_gan_loss"] += beta_gan * gan_loss.item()
        totals["beta_center_loss"] += beta_center * center_loss.item()
        totals["beta_svdd_loss"] += beta_svdd * svdd_loss.item()
        totals["vae_loss"] += vae_loss.item()
        totals["disc_loss"] += disc_loss.item()

    n_batches = max(len(train_loader), 1)
    metrics = {key: value / n_batches for key, value in totals.items()}
    metrics["dis_acc"] = accuracy_score(dis_labels, dis_preds) if dis_labels else 0.0
    metrics["dis_f1"] = f1_score(dis_labels, dis_preds) if dis_labels else 0.0
    return metrics


def validate(encoder, decoder, val_loader, device):
    """Compute validation reconstruction loss."""
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


def train_model(
    config,
    train_loader,
    val_loader,
    train_dataset,
    val_dataset,
    run_dir,
    device,
    training_plotter=None,
    resume_checkpoint=None,
):
    """Train a VAE-GAN model and save checkpoints, curves, and previews."""
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
    beta_center = float(_cfg_get(training_cfg, "beta_center", 0.0))
    beta_svdd = float(_cfg_get(training_cfg, "beta_svdd", 0.0))
    svdd_radius = float(_cfg_get(training_cfg, "svdd_radius", 0.0))
    svdd_nu = float(_cfg_get(training_cfg, "svdd_nu", 0.1))
    save_every = int(_cfg_get(training_cfg, "save_every", 1))
    latent_dim = int(_cfg_get(model_cfg, "latent_dim", 64))

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    encoder = Encoder(z_size=latent_dim).to(device)
    decoder = Decoder(z_size=latent_dim).to(device)
    discriminator = Discriminator().to(device)
    latent_center = torch.zeros(latent_dim, device=device)
    optimizer_enc_dec = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=lr_enc_dec
    )
    optimizer_dis = optim.Adam(discriminator.parameters(), lr=lr_dis)

    dataset_summary = build_dataset_summary(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
    )
    loss_history = []
    best_val_loss = float("inf")
    start_epoch = 0

    if resume_checkpoint is not None:
        checkpoint = load_checkpoint(
            resume_checkpoint,
            encoder=encoder,
            decoder=decoder,
            discriminator=discriminator,
            optimizer_enc_dec=optimizer_enc_dec,
            optimizer_dis=optimizer_dis,
            device=device,
        )
        start_epoch = int(checkpoint.get("epoch") or checkpoint.get("epochs_trained") or 0)
        loss_history = list(checkpoint.get("loss_history") or [])
        validation_losses = [
            row["val_recon_loss"] for row in loss_history if row.get("val_recon_loss") is not None
        ]
        best_val_loss = min(validation_losses, default=float("inf"))
        print(f"[resume] continuing at epoch {start_epoch + 1} of {epochs}")

    for epoch in tqdm(
        range(start_epoch + 1, epochs + 1),
        desc="epochs",
        initial=min(start_epoch, epochs),
        total=epochs,
    ):
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
            beta_center=beta_center,
            beta_svdd=beta_svdd,
            latent_center=latent_center,
            svdd_radius=svdd_radius,
            svdd_nu=svdd_nu,
        )
        val_metrics = validate(encoder, decoder, val_loader, device)
        epoch_metrics = {"epoch": epoch, **train_metrics, **val_metrics}
        loss_history.append(epoch_metrics)

        pd.DataFrame(loss_history).to_csv(run_dir / "loss_history.csv", index=False)
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

        if training_plotter is not None:
            training_plotter.on_epoch_end(
                epoch=epoch,
                total_epochs=epochs,
                loss_history=loss_history,
                encoder=encoder,
                decoder=decoder,
                discriminator=discriminator,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
            )

        message = (
            f"epoch={epoch:04d} "
            f"recon={epoch_metrics['recon_loss']:.6f} "
            f"val_recon={epoch_metrics['val_recon_loss']:.6f} "
            f"disc={epoch_metrics['disc_loss']:.6f}"
        )
        if beta_center > 0.0:
            message += f" center={epoch_metrics['center_loss']:.6f}"
        if beta_svdd > 0.0:
            message += f" svdd={epoch_metrics['svdd_loss']:.6f}"
        print(message)

    return {
        "encoder": encoder,
        "decoder": decoder,
        "discriminator": discriminator,
        "loss_history": loss_history,
        "best_val_loss": best_val_loss,
        "run_dir": run_dir,
    }
