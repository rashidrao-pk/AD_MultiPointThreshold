from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch import optim
from tqdm import tqdm

from .checkpoint import build_dataset_summary, save_checkpoint
from .model import Decoder, Discriminator, Encoder


def _cfg_get(obj, name, default=None):
    """Read an optional config attribute with a default fallback."""
    return getattr(obj, name, default) if obj is not None else default


def _reparameterize(mu, logvar):
    """Sample latent vectors using the VAE reparameterization trick."""
    std = torch.exp(0.5 * logvar)
    return mu + torch.randn_like(std) * std


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
