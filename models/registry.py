from models.vaegan.loader import load_model as load_advis_vaegan_checkpoint
from models.vaegan.trainer import train_model as train_vaegan


MODEL_ALIASES = {
    "advis": "advis_vaegan",
    "advis_vaegan": "advis_vaegan",
    "advis_vae_gan": "advis_vaegan",
    "advis-vae-gan": "advis_vaegan",
    "simple": "simple_vaegan",
    "simple_vaegan": "simple_vaegan",
    "simple_vae_gan": "simple_vaegan",
    "simple-vae-gan": "simple_vaegan",
    "vae_gan": "simple_vaegan",
    "vaegan": "simple_vaegan",
    "vae-gan": "simple_vaegan",
}


MODEL_INFO = {
    "advis_vaegan": {
        "display_name": "ADVIS VAE-GAN",
        "description": "ADVIS/DistriMuSe-compatible VAE-GAN setup for comparison with the original repo.",
        "has_pretrained_loader": True,
    },
    "simple_vaegan": {
        "display_name": "Simple VAE-GAN",
        "description": "Generic VAE-GAN baseline using the shared VAE-GAN trainer.",
        "has_pretrained_loader": False,
    },
}


def normalize_model_name(name):
    key = str(name or "advis_vaegan").lower().replace("-", "_")
    if key not in MODEL_ALIASES:
        raise ValueError(f"Unsupported model: {name}")
    return MODEL_ALIASES[key]


def get_trainer(name):
    model_name = normalize_model_name(name)
    if model_name in {"advis_vaegan", "simple_vaegan"}:
        return train_vaegan
    raise ValueError(f"No trainer registered for model: {name}")


def get_checkpoint_loader(name):
    model_name = normalize_model_name(name)
    if model_name == "advis_vaegan":
        return load_advis_vaegan_checkpoint
    raise ValueError(f"No checkpoint loader registered for model: {name}")


def get_model_info(name):
    return MODEL_INFO[normalize_model_name(name)]


def has_pretrained_loader(name):
    return bool(get_model_info(name).get("has_pretrained_loader", False))
