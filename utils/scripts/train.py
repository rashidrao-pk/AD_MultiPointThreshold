"""
Common training entry point.

This script owns the shared workflow:
- read config
- load dataset
- create a run directory
- dispatch to the selected model trainer
- save the config used for the run

Model-specific loss math lives in models/<model>/trainer.py.
"""

import argparse
import time
from pathlib import Path

import torch

from data import load_data
from utils import read_config, resolve_device, save_config_yaml

def _repo_root():
    """Return the repository root for path resolution."""
    return Path(__file__).resolve().parents[2]


def _namespace_set(obj, name, value):
    """Set an attribute on a namespace-style object."""
    setattr(obj, name, value)


def _resolve_project_paths(config, project_root):
    """Resolve relative data, checkpoint, and output paths against the project root."""
    data_root = Path(config.data.dataset_root)
    if not data_root.is_absolute():
        config.data.dataset_root = str(project_root / data_root)

    checkpoint_root = Path(config.model.checkpoint_root)
    if not checkpoint_root.is_absolute():
        config.model.checkpoint_root = str(project_root / checkpoint_root)

    output_dir = Path(config.output.dir)
    if not output_dir.is_absolute():
        config.output.dir = str(project_root / output_dir)


def _make_run_dir(config, suffix=""):
    """Create a timestamped training run directory."""
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")
    dataset = config.data.name
    category = getattr(config.data, "category", "all")
    model_name = config.model.name
    run_name = f"{dataset}_{category}_{model_name}_{timestamp}"
    if suffix:
        run_name += f"_{suffix}"
    run_dir = Path(config.output.dir) / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def parse_args():
    """Parse command-line arguments for the training entry point."""
    parser = argparse.ArgumentParser(description="Train anomaly-detection models from a config file.")
    parser.add_argument("--config", default="configs/mvtec.yaml", help="Path to YAML config.")
    parser.add_argument("--model", default=None, help="Override config.model.name.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs.")
    parser.add_argument("--device", default=None, help="Override config.device.")
    parser.add_argument("--suffix", default="", help="Optional run-directory suffix.")
    parser.add_argument("--dry_run", action="store_true", help="Load data and create run dir, then stop.")
    return parser.parse_args()


def dispatch_trainer(config):
    """Normalize the model name and return its registered trainer."""
    from models import get_trainer, normalize_model_name

    config.model.name = normalize_model_name(config.model.name)
    return get_trainer(config.model.name)


def main():
    """Run the shared training workflow."""
    args = parse_args()
    project_root = _repo_root()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    config = read_config(config_path)
    if args.model is not None:
        _namespace_set(config.model, "name", args.model)
    if args.device is not None:
        _namespace_set(config, "device", args.device)
    if args.epochs is not None:
        if not hasattr(config, "training"):
            from types import SimpleNamespace

            config.training = SimpleNamespace()
        _namespace_set(config.training, "epochs", args.epochs)

    _resolve_project_paths(config, project_root)
    device = resolve_device(getattr(config, "device", "auto"))
    config.device = str(device)

    print(f"[config] {config_path}")
    print(f"[device] {device}")
    print(f"[data] {config.data.name}/{getattr(config.data, 'category', 'all')}")
    print(f"[model] {config.model.name}")

    train_loader, val_loader, train_dataset, val_dataset = load_data(config)
    print(f"[data] train={len(train_dataset)} val={len(val_dataset)}")

    run_dir = _make_run_dir(config, args.suffix)
    save_config_yaml(config, run_dir / "config.yaml")
    print(f"[run] {run_dir}")

    if args.dry_run:
        print("[dry_run] stopping before trainer dispatch")
        return

    trainer = dispatch_trainer(config)
    result = trainer(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        run_dir=run_dir,
        device=device,
    )

    print(f"[done] run_dir={result['run_dir']}")
    print(f"[done] best_val_loss={result['best_val_loss']:.6f}")


if __name__ == "__main__":
    main()
