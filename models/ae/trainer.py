from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from models.vaegan.checkpoint import build_dataset_summary, save_checkpoint

from .model import Decoder, Encoder


def _cfg_get(obj, name, default=None):
    """Read an optional config attribute with a default fallback."""
    return getattr(obj, name, default) if obj is not None else default


def _reconstruction_loss(name):
    """Return the reconstruction loss function selected for the AE baseline."""
    key = str(name or "MSE").lower()
    if key in {"l1", "mae"}:
        return nn.L1Loss()
    if key in {"mse", "bce", "bcewithlogits"}:
        return nn.MSELoss()
    raise ValueError(f"Unsupported AE reconstruction loss: {name}")


def train_one_epoch(encoder, decoder, train_loader, optimizer, device, loss_fn):
    """Train the autoencoder for one epoch and return averaged metrics."""
    encoder.train()
    decoder.train()
    totals = {
        "recon_loss": 0.0,
        "ae_loss": 0.0,
        "vae_loss": 0.0,
    }

    for images, _ in tqdm(train_loader, desc="train batches", leave=False):
        images = images.to(device, non_blocking=True)

        optimizer.zero_grad()
        z, _ = encoder(images)
        recon = decoder(z)
        recon_loss = loss_fn(recon, images)
        recon_loss.backward()
        optimizer.step()

        totals["recon_loss"] += recon_loss.item()
        totals["ae_loss"] += recon_loss.item()
        totals["vae_loss"] += recon_loss.item()

    n_batches = max(len(train_loader), 1)
    return {key: value / n_batches for key, value in totals.items()}


def validate(encoder, decoder, val_loader, device, loss_fn):
    """Compute validation reconstruction loss for the autoencoder."""
    encoder.eval()
    decoder.eval()
    total = 0.0
    n_batches = 0

    with torch.inference_mode():
        for images, _ in tqdm(val_loader, desc="val batches", leave=False):
            images = images.to(device, non_blocking=True)
            z, _ = encoder(images)
            recon = decoder(z)
            total += loss_fn(recon, images).item()
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
    """Train a basic convolutional autoencoder and save compatible checkpoints."""
    training_cfg = _cfg_get(config, "training", None)
    model_cfg = config.model
    model_name = str(_cfg_get(model_cfg, "name", "basic_ae"))

    epochs = int(_cfg_get(training_cfg, "epochs", 20))
    lr = float(
        _cfg_get(
            training_cfg, "learning_rate_enc_dec", _cfg_get(training_cfg, "learning_rate", 1e-3)
        )
    )
    save_every = int(_cfg_get(training_cfg, "save_every", 1))
    latent_dim = int(_cfg_get(model_cfg, "latent_dim", 64))
    loss_fn = _reconstruction_loss(_cfg_get(model_cfg, "recon_loss", "MSE"))

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    encoder = Encoder(z_size=latent_dim).to(device)
    decoder = Decoder(z_size=latent_dim).to(device)
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)

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
            train_loader,
            optimizer,
            device,
            loss_fn,
        )
        val_metrics = validate(encoder, decoder, val_loader, device, loss_fn)
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
                discriminator=None,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
            )

        if epoch % save_every == 0 or epoch == epochs:
            save_checkpoint(
                run_dir / "model_last.pt",
                encoder,
                decoder,
                None,
                optimizer_enc_dec=optimizer,
                epoch=epoch,
                loss_history=loss_history,
                config=config,
                dataset_summary=dataset_summary,
                metrics=epoch_metrics,
                model_name=model_name,
                notes="Latest basic autoencoder checkpoint for this training run.",
            )

        if val_metrics["val_recon_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_recon_loss"]
            save_checkpoint(
                run_dir / "model_best.pt",
                encoder,
                decoder,
                None,
                optimizer_enc_dec=optimizer,
                epoch=epoch,
                loss_history=loss_history,
                config=config,
                dataset_summary=dataset_summary,
                metrics=epoch_metrics,
                model_name=model_name,
                notes="Best basic autoencoder checkpoint by validation reconstruction loss.",
            )

        print(
            f"epoch={epoch:04d} "
            f"recon={epoch_metrics['recon_loss']:.6f} "
            f"val_recon={epoch_metrics['val_recon_loss']:.6f}"
        )

    return {
        "encoder": encoder,
        "decoder": decoder,
        "discriminator": None,
        "loss_history": loss_history,
        "best_val_loss": best_val_loss,
        "run_dir": run_dir,
    }
