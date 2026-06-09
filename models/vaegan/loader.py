from pathlib import Path
import torch
from .model import Encoder, Decoder, Discriminator

def get_checkpoint_path(model_cfg, data_cfg):
    # i think they all have this format, to check... 
    if data_cfg.name == "MVTec":
        dataset_folder = "AD_MVTec"
        ckpt_name = f"model_{data_cfg.category}_{model_cfg.latent_dim}.pt"

    elif data_cfg.name == "Cobots_Synthetic":
        dataset_folder = "AD_Cobots_Synthetic"
        ckpt_name = f"model_{data_cfg.category}_{model_cfg.latent_dim}.pt"

    elif data_cfg.name == "Robotics_Hazards":
        dataset_folder = "AD_Robotics_Hazards"
        ckpt_name = f"model_{data_cfg.category}_{model_cfg.latent_dim}.pt"

    else:
        raise ValueError(f"Unknown dataset name: {data_cfg.name}")

    return Path(model_cfg.checkpoint_root) / dataset_folder / "checkpoints" / ckpt_name


def load_model(config, device):
    model_cfg = config.model
    data_cfg = config.data
    ckpt_path = get_checkpoint_path(model_cfg, data_cfg)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    encoder = Encoder(z_size=model_cfg.latent_dim).to(device)
    decoder = Decoder(z_size=model_cfg.latent_dim).to(device)
    discriminator = Discriminator().to(device)
    
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

    return encoder, decoder, discriminator