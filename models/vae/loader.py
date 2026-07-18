import torch

from models.vaegan.loader import get_checkpoint_path
from models.vaegan.model import Decoder, Encoder


def load_model(config, device):
    """Load vanilla VAE encoder and decoder weights for inference."""
    model_cfg = config.model
    data_cfg = config.data
    ckpt_path = get_checkpoint_path(model_cfg, data_cfg)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    encoder = Encoder(z_size=model_cfg.latent_dim).to(device)
    decoder = Decoder(z_size=model_cfg.latent_dim).to(device)

    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])

    return encoder, decoder, None
