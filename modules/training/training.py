"""
Common training workflow.

This module owns the shared workflow:
- read config
- detect existing inference checkpoints
- load dataset
- create a run directory
- dispatch to the selected model trainer
- publish the best checkpoint
- save the config used for the run

Model-specific loss math lives in models/<model>/trainer.py.
"""

import argparse
import copy
import csv
import hashlib
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from data import load_data
from models.vaegan.loader import get_checkpoint_path
from utils import apply_config_overrides, read_config, resolve_device, save_config_yaml
from utils.general import namespace_to_dict


TRAIN_RUNS_CSV = "training_runs.csv"

DATASET_ALIASES = {
    "mvtec": "MVTec",
    "mvtex": "MVTec",
    "mvtec_ad": "MVTec",
    "mvtec-ad": "MVTec",
    "mvtec_anomaly_detection": "MVTec",
    "cobot": "Cobots_Synthetic",
    "cobots": "Cobots_Synthetic",
    "cobots_synthetic": "Cobots_Synthetic",
    "distrimuse": "Cobots_Synthetic",
    "distrimuse_unigra": "Cobots_Synthetic",
    "robotics_hazards": "Robotics_Hazards",
    "robotics-hazards": "Robotics_Hazards",
    "hazards": "Robotics_Hazards",
    "corridor": "Robotics_Hazards",
}

DATASET_CATEGORIES = {
    "MVTec": [
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    ],
    "Cobots_Synthetic": ["PLeft", "PRight", "ConvBelt", "RoboArm"],
    "Robotics_Hazards": ["corridor"],
}


def _repo_root():
    """Return the repository root for path resolution."""
    return Path(__file__).resolve().parents[2]


def _namespace_set(obj, name, value):
    """Set an attribute on a namespace-style object."""
    setattr(obj, name, value)


def _normalize_dataset_name(dataset):
    """Return the canonical dataset name used by configs and data loaders."""
    if dataset is None:
        return None

    key = str(dataset).strip()
    return DATASET_ALIASES.get(key.lower(), key)


def _known_categories(dataset):
    """Return known categories/areas for a supported dataset."""
    dataset = _normalize_dataset_name(dataset)
    if dataset not in DATASET_CATEGORIES:
        raise ValueError(
            f"Unknown dataset for training sweep: {dataset}. "
            f"Supported: {', '.join(DATASET_CATEGORIES)}"
        )
    return DATASET_CATEGORIES[dataset]


def _resolve_config_path(config_arg, project_root):
    """Resolve a user-supplied config path relative to the project root."""
    config_path = Path(config_arg)
    if not config_path.exists() and "\\" in str(config_arg):
        config_path = Path(str(config_arg).replace("\\", "/"))
    if not config_path.is_absolute():
        config_path = project_root / config_path
    return config_path


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


def _training_signature(config, config_path, suffix=""):
    """Create a stable hash for the training inputs that define a run."""
    payload = {
        "data": namespace_to_dict(config.data),
        "model": namespace_to_dict(config.model),
        "training": namespace_to_dict(getattr(config, "training", SimpleNamespace())),
        "suffix": suffix,
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _training_match_payload(config):
    """Return the fields that define whether a model is trained for this data."""
    return {
        "data": namespace_to_dict(config.data),
        "model": namespace_to_dict(config.model),
        "training": namespace_to_dict(getattr(config, "training", SimpleNamespace())),
    }


def _saved_training_match_payload(saved_config):
    """Return comparable training fields from a saved run config dictionary."""
    return {
        "data": saved_config.get("data", {}),
        "model": saved_config.get("model", {}),
        "training": saved_config.get("training", {}),
    }


def _count_epochs(run_dir):
    """Return the number of saved training epochs for a run directory."""
    for path in (
        run_dir / "loss_history.csv",
        run_dir / "model_best_loss_history.csv",
        run_dir / "model_last_loss_history.csv",
    ):
        if path.exists():
            with open(path, "r", newline="", encoding="utf-8") as handle:
                return max(sum(1 for _ in csv.DictReader(handle)), 0)

    for path in (run_dir / "model_best_config.json", run_dir / "model_last_config.json"):
        if path.exists():
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle).get("epochs_trained", "")

    return ""


def _inference_checkpoint_path(config):
    """Return the checkpoint path used by inference for this config."""
    return get_checkpoint_path(config.model, config.data)


def _forced_checkpoint_path(config, checkpoint_path):
    """Return the new-checkpoints path that mirrors the configured hierarchy."""
    checkpoint_root = Path(config.model.checkpoint_root)
    try:
        relative_checkpoint = checkpoint_path.relative_to(checkpoint_root)
    except ValueError:
        relative_checkpoint = checkpoint_path.name
    return checkpoint_root / "new_checkpoints" / relative_checkpoint


def _checkpoint_epochs(checkpoint):
    """Return the number of training epochs recorded in a checkpoint."""
    if checkpoint.get("epochs_trained") is not None:
        return checkpoint["epochs_trained"]

    if checkpoint.get("epoch") is not None:
        return checkpoint["epoch"]

    loss_history = checkpoint.get("loss_history")
    if loss_history is None:
        return ""

    try:
        return len(loss_history)
    except TypeError:
        return ""


def _read_checkpoint_info(checkpoint_path):
    """Read lightweight metadata from an inference checkpoint."""
    if not checkpoint_path.exists():
        return None

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return {
        "checkpoint_path": str(checkpoint_path),
        "epochs": _checkpoint_epochs(checkpoint),
        "has_encoder": "encoder_state_dict" in checkpoint,
        "has_decoder": "decoder_state_dict" in checkpoint,
        "has_discriminator": "discriminator_state_dict" in checkpoint,
    }


def _print_existing_checkpoint(config, checkpoint_info):
    """Print a human-readable message for an already trained inference checkpoint."""
    print("-" * 80)
    print("[skip] Model checkpoint already exists for this config/data.")
    print(f"[skip] Dataset: {config.data.name}/{getattr(config.data, 'category', '')}")
    print(f"[skip] Model: {config.model.name}")
    print(f"[skip] Epochs trained: {checkpoint_info.get('epochs', '')}")
    print(f"[skip] Inference checkpoint: {checkpoint_info['checkpoint_path']}")
    print("-" * 80)
    print("[skip] Use --force or --force_train to train again.")


def _append_training_run(csv_path, row):
    """Append a training run row while preserving previous CSV columns."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists():
        with open(csv_path, "r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            old_fieldnames = reader.fieldnames or []
            old_rows = list(reader)
        fieldnames = list(dict.fromkeys(old_fieldnames + list(row.keys())))
    else:
        old_rows = []
        fieldnames = list(row.keys())

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(old_rows)
        writer.writerow(row)


def _publish_best_checkpoint(run_dir, checkpoint_path):
    """Copy the best training checkpoint to the selected checkpoint path."""
    best_path = Path(run_dir) / "model_best.pt"
    if not best_path.exists():
        best_path = Path(run_dir) / "model_last.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"No trained checkpoint found in run directory: {run_dir}")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_path, checkpoint_path)
    return checkpoint_path


def parse_args():
    """Parse command-line arguments for the training entry point."""
    parser = argparse.ArgumentParser(description="Train anomaly-detection models from a config file.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--model",
        default=None,
        help="Override config.model.name. Use 'all' to train every registered trainable model.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs.")
    parser.add_argument("--device", default=None, help="Override config.device.")
    parser.add_argument("--suffix", default="", help="Optional run-directory suffix.")
    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "Override data.name and train every known category for that dataset "
            "unless --category selects one category. Examples: MVTec, Cobots_Synthetic."
        ),
    )
    parser.add_argument(
        "--category",
        default=None,
        help=(
            "Override data.category, for example --category zipper. "
            "Use --category all with --dataset to train all known categories."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Override data.dataset_root.",
    )
    parser.add_argument(
        "--checkpoint-root",
        default=None,
        help="Override model.checkpoint_root.",
    )
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override any config value using dotted keys. "
            "Can be repeated, e.g. --set data.category=zipper "
            "--set training.beta_center=0.0001."
        ),
    )
    parser.add_argument(
        "--force",
        "--force_train",
        action="store_true",
        help="Train again even if the same config/model/data run already completed.",
    )
    parser.add_argument(
        "--plot_curves",
        action="store_true",
        help="Save training-curve plots during training.",
    )
    parser.add_argument(
        "--plot_every",
        type=int,
        default=10,
        help="Save selected training plots every N epochs.",
    )
    parser.add_argument(
        "--plot_latent_space",
        action="store_true",
        help="Save latent-space plots during training.",
    )
    parser.add_argument(
        "--plot_score_distribution",
        action="store_true",
        help="Save reconstruction-score distribution plots during training.",
    )
    parser.add_argument(
        "--plot_quality",
        action="store_true",
        help="Save training-quality plots such as AUROC, score components, latent radius, and validation grids.",
    )
    parser.add_argument(
        "--latent_space_classes",
        choices=["normal", "both"],
        default="normal",
        help="Choose whether latent-space plots use normal training data only or both validation/test classes.",
    )
    parser.add_argument(
        "--latent_projection",
        choices=["tsne", "pca"],
        default="tsne",
        help="Projection method for latent-space plots.",
    )
    parser.add_argument(
        "--latent_plot_batches",
        type=int,
        default=8,
        help="Maximum batches to encode for each latent-space plot.",
    )
    parser.add_argument(
        "--score_plot_batches",
        type=int,
        default=8,
        help="Maximum train/validation batches to score for each score-distribution plot.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Load data and create run dir, then stop.")
    return parser.parse_args()


def _config_overrides_from_args(args):
    """Collect config overrides from friendly CLI flags and generic --set values."""
    overrides = list(args.set_overrides)

    if args.dataset is not None:
        overrides.append(f"data.name={_normalize_dataset_name(args.dataset)}")
    if args.category is not None and str(args.category).lower() != "all":
        overrides.append(f"data.category={args.category}")
    if args.dataset_root is not None:
        overrides.append(f"data.dataset_root={args.dataset_root}")
    if args.checkpoint_root is not None:
        overrides.append(f"model.checkpoint_root={args.checkpoint_root}")
    if args.device is not None:
        overrides.append(f"device={args.device}")
    if args.epochs is not None:
        overrides.append(f"training.epochs={args.epochs}")

    return overrides


def _category_sweep(config, args):
    """Return categories to sweep, or an empty list for single-category training."""
    category = str(args.category).lower() if args.category is not None else None

    if category == "all":
        return _known_categories(args.dataset or config.data.name)

    if args.dataset is not None and args.category is None:
        return _known_categories(args.dataset)

    return []


def dispatch_trainer(config):
    """Normalize the model name and return its registered trainer."""
    from models import get_trainer, normalize_model_name

    config.model.name = normalize_model_name(config.model.name)
    return get_trainer(config.model.name)


def _selected_model_names(config, model_arg):
    """Return canonical model names selected by the command-line arguments."""
    from models import list_trainable_models, normalize_model_name

    if model_arg is None:
        return [normalize_model_name(config.model.name)]

    if str(model_arg).lower() == "all":
        model_names = list_trainable_models()
        if not model_names:
            raise ValueError("No trainable models are registered.")
        return model_names

    return [normalize_model_name(model_arg)]


def _run_training_for_model(args, project_root, config_path, base_config, model_name):
    """Run the shared workflow for one concrete model name."""
    config = copy.deepcopy(base_config)
    _namespace_set(config.model, "name", model_name)
    if args.device is not None:
        _namespace_set(config, "device", args.device)
    if args.epochs is not None:
        if not hasattr(config, "training"):
            config.training = SimpleNamespace()
        _namespace_set(config.training, "epochs", args.epochs)

    _resolve_project_paths(config, project_root)
    device = resolve_device(getattr(config, "device", "auto"))
    config.device = str(device)
    trainer = dispatch_trainer(config)

    checkpoint_path = _inference_checkpoint_path(config)
    publish_checkpoint_path = (
        _forced_checkpoint_path(config, checkpoint_path)
        if args.force
        else checkpoint_path
    )
    checkpoint_info = _read_checkpoint_info(checkpoint_path)
    if checkpoint_info is not None and not args.force:
        _print_existing_checkpoint(config, checkpoint_info)
        return None

    runs_csv_path = Path(config.output.dir) / TRAIN_RUNS_CSV
    run_hash = _training_signature(config, config_path, args.suffix)

    print(f"[config] {config_path}")
    print(f"[device] {device}")
    print(f"[data] {config.data.name}/{getattr(config.data, 'category', 'all')}")
    print(f"[model] {config.model.name}")
    print(f"[checkpoint] {checkpoint_path}")
    if args.force:
        print(f"[new_checkpoint] {publish_checkpoint_path}")
    print("-" * 80)

    train_loader, val_loader, train_dataset, val_dataset = load_data(config)
    print(f"[data] train={len(train_dataset)} val={len(val_dataset)}")

    run_dir = _make_run_dir(config, args.suffix)
    save_config_yaml(config, run_dir / "config.yaml")
    print(f"[run] {run_dir}")
    print(f"[hash] {run_hash}")

    if args.dry_run:
        print("[dry_run] stopping before trainer dispatch")
        return None

    training_plotter = None
    if args.plot_curves or args.plot_latent_space or args.plot_score_distribution or args.plot_quality:
        from modules.plotting import TrainingPlotter

        training_plotter = TrainingPlotter(
            run_dir=run_dir,
            plot_curves=args.plot_curves,
            plot_latent_space=args.plot_latent_space,
            plot_score_distribution=args.plot_score_distribution,
            plot_quality=args.plot_quality,
            latent_space_classes=args.latent_space_classes,
            latent_projection=args.latent_projection,
            plot_every=args.plot_every,
            max_latent_batches=args.latent_plot_batches,
            max_score_batches=args.score_plot_batches,
        )
        print(
            "[plots] "
            f"curves={args.plot_curves} every={args.plot_every} "
            f"latent={args.plot_latent_space} "
            f"latent_classes={args.latent_space_classes} "
            f"latent_projection={args.latent_projection} "
            f"score_distribution={args.plot_score_distribution} "
            f"quality={args.plot_quality}"
        )

    result = trainer(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        run_dir=run_dir,
        device=device,
        training_plotter=training_plotter,
    )

    published_checkpoint = _publish_best_checkpoint(result["run_dir"], publish_checkpoint_path)

    _append_training_run(
        runs_csv_path,
        {
            "hash": run_hash,
            "status": "done",
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config_path": str(config_path),
            "run_dir": str(result["run_dir"]),
            "dataset": config.data.name,
            "category": getattr(config.data, "category", ""),
            "model": config.model.name,
            "epochs": len(result.get("loss_history", [])),
            "best_val_loss": result.get("best_val_loss", ""),
            "suffix": args.suffix,
            "forced": bool(args.force),
            "model_best_path": str(Path(result["run_dir"]) / "model_best.pt"),
            "model_last_path": str(Path(result["run_dir"]) / "model_last.pt"),
            "inference_checkpoint": str(checkpoint_path),
            "published_checkpoint": str(published_checkpoint),
        },
    )

    print(f"[done] run_dir={result['run_dir']}")
    print(f"[done] published_checkpoint={published_checkpoint}")
    print(f"[done] best_val_loss={result['best_val_loss']:.6f}")
    return result


def main():
    """Run the shared training workflow."""
    args = parse_args()
    project_root = _repo_root()
    config_path = _resolve_config_path(args.config, project_root)

    base_config = read_config(config_path)
    applied_overrides = apply_config_overrides(base_config, _config_overrides_from_args(args))
    for key, value in applied_overrides:
        print(f"[config_override] {key} = {value}")

    model_names = _selected_model_names(base_config, args.model)
    print(f"[models] selected={', '.join(model_names)}")

    categories = _category_sweep(base_config, args)
    if categories:
        total = len(categories) * len(model_names)
        completed = 0
        for category in categories:
            category_config = copy.deepcopy(base_config)
            _namespace_set(category_config.data, "category", category)
            print("=" * 80)
            print(f"[sweep] dataset={category_config.data.name} category={category}")
            print("=" * 80)
            for model_name in model_names:
                _run_training_for_model(args, project_root, config_path, category_config, model_name)
                completed += 1
        print("=" * 80)
        print(f"[sweep] completed {completed}/{total} training jobs")
        print("=" * 80)
        return

    for model_name in model_names:
        _run_training_for_model(args, project_root, config_path, base_config, model_name)


if __name__ == "__main__":
    main()
