"""
Model Loader and Inspector
===========================
Robust utility to list, load, and test VAE-GAN checkpoints across CPU, CUDA,
and Apple Silicon MPS.

Examples:
    python scripts/model_loader.py --list
    python scripts/model_loader.py --checkpoint models --list
    python scripts/model_loader.py --model_path path/to/model.pt --device auto --test_forward
    python scripts/model_loader.py --checkpoint models --safety_area RoboArm --show_summary
"""

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import torch

import utils.scripts.utils_model as utmc
from utils.scripts.utils_model import Encoder, Decoder, Discriminator


ENCODER_KEYS = ("encoder_state_dict", "encoder", "Enc", "enc")
DECODER_KEYS = ("decoder_state_dict", "decoder", "Dec", "dec")
DISCRIMINATOR_KEYS = ("discriminator_state_dict", "discriminator", "Dis", "dis")


def resolve_device(requested: str = "auto") -> torch.device:
    """Return the best available device: CUDA > MPS > CPU, unless explicitly set."""
    requested = (requested or "auto").lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print("[device] CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    if requested == "mps":
        mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        if not mps_ok:
            print("[device] MPS requested but not available. Falling back to CPU.")
            return torch.device("cpu")
    return torch.device(requested)


def torch_load_checkpoint(model_path: Path, device: torch.device) -> Dict[str, Any]:
    """Load checkpoint safely. Load on CPU first; model is moved to device afterwards."""
    try:
        return torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(model_path, map_location="cpu")


def _first_existing_state_dict(checkpoint: Dict[str, Any], keys) -> Optional[Dict[str, torch.Tensor]]:
    """Return the first checkpoint state dict found under one of the given keys."""
    for key in keys:
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]
    return None


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Handle checkpoints saved from DataParallel/DistributedDataParallel."""
    out = {}
    for k, v in state_dict.items():
        out[k[7:] if k.startswith("module.") else k] = v
    return out


def count_parameters(model):
    """Return total and trainable parameter counts for a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model_summary(model, model_name="Model"):
    """Return a printable architecture and parameter-count summary."""
    total_params, trainable_params = count_parameters(model)
    return f"""
{'='*80}
{model_name}
{'='*80}
Total Parameters:      {total_params:,}
Trainable Parameters:  {trainable_params:,}
{'='*80}
Architecture:
{model}
{'='*80}
"""


def instantiate_models(latent_dims: int, device: torch.device) -> Dict[str, torch.nn.Module]:
    """Instantiate encoder, decoder, and discriminator modules on a device."""
    return {
        "encoder": Encoder(z_size=latent_dims).to(device).eval(),
        "decoder": Decoder(z_size=latent_dims).to(device).eval(),
        "discriminator": Discriminator().to(device).eval(),
    }


def load_all_models(model_path, latent_dims=64, device="auto", strict=True):
    """Load encoder, decoder, discriminator from one VAE-GAN checkpoint."""
    model_path = Path(model_path)
    device = resolve_device(str(device)) if not isinstance(device, torch.device) else device

    print(f"\nLoading checkpoint: {model_path}")
    print(f"Device: {device}")
    print(f"Latent dimensions: {latent_dims}\n")

    checkpoint = torch_load_checkpoint(model_path, device)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")

    enc_sd = _first_existing_state_dict(checkpoint, ENCODER_KEYS)
    dec_sd = _first_existing_state_dict(checkpoint, DECODER_KEYS)
    dis_sd = _first_existing_state_dict(checkpoint, DISCRIMINATOR_KEYS)

    if enc_sd is None or dec_sd is None:
        available = ", ".join(checkpoint.keys())
        raise KeyError(
            "Checkpoint does not contain expected VAE-GAN keys. "
            f"Need encoder/decoder keys. Available keys: {available}"
        )

    models = instantiate_models(latent_dims, device)
    models["encoder"].load_state_dict(_strip_module_prefix(enc_sd), strict=strict)
    models["decoder"].load_state_dict(_strip_module_prefix(dec_sd), strict=strict)

    if dis_sd is not None:
        models["discriminator"].load_state_dict(_strip_module_prefix(dis_sd), strict=strict)
        print("✓ Loaded discriminator")
    else:
        models["discriminator"] = None
        print("! Discriminator state not found; loaded encoder/decoder only")

    print("✓ Loaded encoder")
    print("✓ Loaded decoder")
    return models


def list_available_models(checkpoint_dir):
    """Print and return checkpoints found under a directory."""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return []

    models = sorted(checkpoint_path.rglob("*.pt")) + sorted(checkpoint_path.rglob("*.pth"))
    if not models:
        print(f"No .pt/.pth checkpoints found in {checkpoint_dir}")
        return []

    print(f"\nAvailable checkpoints in {checkpoint_dir}:")
    print(f"{'#':<4} {'Path':<80} {'Size (MB)':>10}")
    print("-" * 100)
    for idx, path in enumerate(models, 1):
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"{idx:<4} {str(path):<80} {size_mb:>10.2f}")
    return models


def display_model_details(model_dict):
    """Print model summaries for loaded model objects."""
    for model_name, model in model_dict.items():
        if model is not None:
            print(get_model_summary(model, model_name=model_name.upper()))


def test_model_forward_pass(models_dict, input_shape=(1, 3, 128, 128), device="auto"):
    """Run a synthetic forward pass through loaded VAE-GAN modules."""
    device = resolve_device(str(device)) if not isinstance(device, torch.device) else device
    x = torch.randn(input_shape, device=device)
    print(f"\nForward pass test on {device}; input shape: {tuple(x.shape)}")
    with torch.no_grad():
        mu, logvar = models_dict["encoder"](x)
        z = utmc.reparameterize(mu, logvar)
        recon = models_dict["decoder"](z)
        print(f"✓ Encoder: mu={tuple(mu.shape)}, logvar={tuple(logvar.shape)}")
        print(f"✓ Decoder: recon={tuple(recon.shape)}")
        if models_dict.get("discriminator") is not None:
            pred_real = models_dict["discriminator"](x)
            pred_fake = models_dict["discriminator"](recon)
            print(f"✓ Discriminator: real={tuple(pred_real.shape)}, fake={tuple(pred_fake.shape)}")
    print("✓ Forward pass successful")


def get_checkpoint_info(checkpoint_path):
    """Return basic metadata and key availability for a checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch_load_checkpoint(checkpoint_path, torch.device("cpu"))
    if not isinstance(checkpoint, dict):
        return {"checkpoint_path": str(checkpoint_path), "type": str(type(checkpoint))}
    return {
        "checkpoint_path": str(checkpoint_path),
        "file_size_mb": round(checkpoint_path.stat().st_size / (1024 * 1024), 2),
        "keys": list(checkpoint.keys()),
        "has_encoder": _first_existing_state_dict(checkpoint, ENCODER_KEYS) is not None,
        "has_decoder": _first_existing_state_dict(checkpoint, DECODER_KEYS) is not None,
        "has_discriminator": _first_existing_state_dict(checkpoint, DISCRIMINATOR_KEYS) is not None,
        "has_loss_history": "loss_history" in checkpoint,
        "has_config": "config" in checkpoint,
    }


def parse_args():
    """Parse command-line arguments for model inspection."""
    p = argparse.ArgumentParser(description="Load and inspect trained VAE-GAN models")
    p.add_argument("--list", action="store_true", help="List all checkpoints recursively")
    p.add_argument("--checkpoint", default="models", help="Checkpoint directory")
    p.add_argument("--model_path", type=str, help="Path to specific checkpoint")
    p.add_argument("--safety_area", default="RoboArm", help="Safety area name used to find checkpoint")
    p.add_argument("--latent_dims", type=int, default=64, help="Latent dimensions")
    p.add_argument("--show_summary", action="store_true", help="Show model summaries")
    p.add_argument("--test_forward", action="store_true", help="Test forward pass")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Device")
    p.add_argument("--non_strict", action="store_true", help="Load with strict=False")
    p.add_argument("--verbose", "-v", action="store_true", help="Show checkpoint metadata")
    return p.parse_args()


def main():
    """Run the model loader and inspector command."""
    args = parse_args()
    device = resolve_device(args.device)
    print(f"[device] Using: {device}")

    if args.list:
        list_available_models(args.checkpoint)
        return

    if args.model_path:
        model_path = Path(args.model_path)
    else:
        candidates = sorted(Path(args.checkpoint).rglob(f"*{args.safety_area}*.pt"))
        if not candidates:
            print(f"No checkpoint found for safety area: {args.safety_area}")
            list_available_models(args.checkpoint)
            return
        model_path = candidates[0]
        print(f"Found checkpoint: {model_path}")

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    try:
        models = load_all_models(model_path, args.latent_dims, device, strict=not args.non_strict)
    except Exception as e:
        print(f"✗ Failed to load checkpoint: {e}")
        return

    if args.show_summary:
        display_model_details(models)
    if args.test_forward:
        test_model_forward_pass(models, device=device)
    if args.verbose:
        print("\nCheckpoint info:")
        for k, v in get_checkpoint_info(model_path).items():
            print(f"  {k}: {v}")

    print("\n✓ Model inspection complete")


if __name__ == "__main__":
    main()
