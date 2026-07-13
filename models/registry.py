from models.vaegan.loader import load_model as load_advis_vaegan_checkpoint
from models.vaegan.trainer import train_model as train_vaegan
from models.ae.loader import load_model as load_basic_ae_checkpoint
from models.ae.trainer import train_model as train_basic_ae


MODEL_ALIASES = {
    "advis": "advis_vaegan",
    "advis_vaegan": "advis_vaegan",
    "advis_vae_gan": "advis_vaegan",
    "advis-vae-gan": "advis_vaegan",
    "vae_gan": "advis_vaegan",
    "vaegan": "advis_vaegan",
    "vae-gan": "advis_vaegan",
    "simple": "simple_vaegan",
    "simple_vaegan": "simple_vaegan",
    "simple_vae_gan": "simple_vaegan",
    "simple-vae-gan": "simple_vaegan",
    "ae": "basic_ae",
    "basic_ae": "basic_ae",
    "autoencoder": "basic_ae",
    "basic_autoencoder": "basic_ae",
}


MODEL_INFO = {
    "advis_vaegan": {
        "display_name": "ADVIS VAE-GAN",
        "description": "ADVIS/DistriMuSe-compatible VAE-GAN setup for comparison with the original repo.",
        "has_pretrained_loader": True,
        "trainable": True,
    },
    "simple_vaegan": {
        "display_name": "Simple VAE-GAN",
        "description": "Generic VAE-GAN baseline using the shared VAE-GAN trainer.",
        "has_pretrained_loader": False,
        "trainable": False,
    },
    "basic_ae": {
        "display_name": "Basic AE",
        "description": "Convolutional autoencoder reconstruction baseline with no KL or GAN loss.",
        "has_pretrained_loader": True,
        "trainable": True,
    },
}


def normalize_model_name(name):
    """Return the canonical registry name for a configured model alias."""
    key = str(name or "advis_vaegan").lower().replace("-", "_")
    if key not in MODEL_ALIASES:
        raise ValueError(f"Unsupported model: {name}")
    return MODEL_ALIASES[key]


def get_trainer(name):
    """Return the training function registered for a model name."""
    model_name = normalize_model_name(name)
    if model_name in {"advis_vaegan", "simple_vaegan"}:
        return train_vaegan
    if model_name == "basic_ae":
        return train_basic_ae
    raise ValueError(f"No trainer registered for model: {name}")


def list_trainable_models():
    """Return canonical model names that should run for --model all."""
    return [name for name, info in MODEL_INFO.items() if info.get("trainable")]


def get_checkpoint_loader(name):
    """Return the checkpoint-loading function registered for a model name."""
    model_name = normalize_model_name(name)
    if model_name == "advis_vaegan":
        return load_advis_vaegan_checkpoint
    if model_name == "basic_ae":
        return load_basic_ae_checkpoint
    raise ValueError(f"No checkpoint loader registered for model: {name}")


def get_model_info(name):
    """Return display metadata for a registered model."""
    return MODEL_INFO[normalize_model_name(name)]


def has_pretrained_loader(name):
    """Return whether a model has a registered pretrained checkpoint loader."""
    return bool(get_model_info(name).get("has_pretrained_loader", False))
