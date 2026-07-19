from pathlib import Path
import torch
from .model import Encoder, Decoder, Discriminator


def get_checkpoint_path(model_cfg, data_cfg):
    """Build the expected checkpoint path for a dataset/model configuration."""
    model_name = str(getattr(model_cfg, "name", "advis_vaegan")).lower().replace("-", "_")
    if model_name in {"ae", "autoencoder", "basic_autoencoder"}:
        model_name = "basic_ae"
    if model_name in {"vae", "variational_autoencoder"}:
        model_name = "vanilla_vae"
    # i think they all have this format, to check...
    if data_cfg.name == "MVTec":
        dataset_folder = "AD_MVTec"

    elif data_cfg.name == "Cobots_Synthetic":
        dataset_folder = "AD_Cobots_Synthetic"

    elif data_cfg.name == "Robotics_Hazards":
        dataset_folder = "AD_Robotics_Hazards"

    else:
        raise ValueError(f"Unknown dataset name: {data_cfg.name}")

    if model_name in {
        "advis",
        "advis_vaegan",
        "advis_vae_gan",
        "vae_gan",
        "vaegan",
        "simple_vaegan",
    }:
        ckpt_name = f"model_{data_cfg.category}_{model_cfg.latent_dim}.pt"
    else:
        ckpt_name = f"model_{model_name}_{data_cfg.category}_{model_cfg.latent_dim}.pt"

    return Path(model_cfg.checkpoint_root) / dataset_folder / "checkpoints" / ckpt_name


def load_model(config, device):
    """Load VAE-GAN encoder, decoder, and discriminator weights for inference."""
    model_cfg = config.model
    model_name = str(getattr(model_cfg, "name", "advis_vaegan")).lower().replace("-", "_")
    if model_name in {"ae", "basic_ae", "autoencoder", "basic_autoencoder"}:
        from models.ae.loader import load_model as load_ae_model

        return load_ae_model(config, device)
    if model_name in {"vae", "vanilla_vae", "variational_autoencoder"}:
        from models.vae.loader import load_model as load_vae_model

        return load_vae_model(config, device)

    data_cfg = config.data
    ckpt_path = get_checkpoint_path(model_cfg, data_cfg)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    encoder = Encoder(z_size=model_cfg.latent_dim).to(device)
    decoder = Decoder(z_size=model_cfg.latent_dim).to(device)
    discriminator = Discriminator().to(device)

    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"])

    return encoder, decoder, discriminator
