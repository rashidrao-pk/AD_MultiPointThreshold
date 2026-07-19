from pathlib import Path
import json
import os
import sys

os.environ.setdefault(
    "XDG_CACHE_HOME",
    str(Path(__file__).resolve().parents[1] / ".cache"),
)
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"),
)

import matplotlib.pyplot as plt
import pandas as pd
import torch
from IPython.display import display
from torch.utils.data import DataLoader, Subset


def setup_project():
    """Resolve the project root and ensure it is importable."""
    project_root = Path.cwd()
    if project_root.name == "notebooks":
        project_root = project_root.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


def load_config_data(config_name):
    """Load a config, dataset objects, loaders, and metadata for verification notebooks."""
    project_root = setup_project()

    from data import load_data
    from data.metadata import dataset_summary, sample_records
    from utils import read_config, resolve_device

    config_path = project_root / "configs" / config_name
    cfg = read_config(config_path)

    if not Path(cfg.data.dataset_root).is_absolute():
        cfg.data.dataset_root = str(project_root / cfg.data.dataset_root)
    if not Path(cfg.model.checkpoint_root).is_absolute():
        cfg.model.checkpoint_root = str(project_root / cfg.model.checkpoint_root)
    if not Path(cfg.output.dir).is_absolute():
        cfg.output.dir = str(project_root / cfg.output.dir)

    device = resolve_device(getattr(cfg, "device", "auto"))
    cfg.device = str(device)

    train_loader, test_loader, train_dataset, test_dataset = load_data(cfg)
    summary = {
        "train": dataset_summary(train_dataset, "train"),
        "test": dataset_summary(test_dataset, "test"),
    }

    return {
        "project_root": project_root,
        "config_path": config_path,
        "cfg": cfg,
        "device": device,
        "train_loader": train_loader,
        "test_loader": test_loader,
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "summary": summary,
        "sample_records": sample_records,
    }


def subset_indices(dataset, max_samples=None):
    """Select a class-balanced subset of dataset indices when possible."""
    if max_samples is None or max_samples >= len(dataset):
        return list(range(len(dataset)))

    samples = getattr(dataset, "samples", None)
    if not samples:
        return list(range(min(max_samples, len(dataset))))

    by_label = {}
    for idx, (_, label) in enumerate(samples):
        by_label.setdefault(int(label), []).append(idx)

    n_classes = max(len(by_label), 1)
    per_class = max(max_samples // n_classes, 1)
    selected = []
    for label in sorted(by_label):
        selected.extend(by_label[label][:per_class])

    remaining = max_samples - len(selected)
    if remaining > 0:
        selected_set = set(selected)
        selected.extend(idx for idx in range(len(dataset)) if idx not in selected_set)
        selected = selected[:max_samples]

    return sorted(selected[:max_samples])


def subset_loader(dataset, batch_size, max_samples=None, shuffle=False, num_workers=0):
    """Build a DataLoader over a full dataset or selected subset."""
    if max_samples is not None:
        dataset = Subset(dataset, subset_indices(dataset, max_samples))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def denorm(image_tensor):
    """Convert a normalized CHW tensor into an HWC image array."""
    return ((image_tensor.detach().cpu().clamp(-1, 1) + 1) / 2).permute(1, 2, 0).numpy()


def show_samples(dataset, title, max_images=8):
    """Display sample images and return their metadata records."""
    records = []
    images = []
    for idx in range(min(max_images, len(dataset))):
        image, label = dataset[idx]
        images.append(image)
        path, raw_label = dataset.samples[idx]
        records.append((Path(path), int(raw_label), int(label)))

    fig, axes = plt.subplots(1, len(images), figsize=(2.4 * len(images), 2.8))
    if len(images) == 1:
        axes = [axes]
    for ax, image, (path, raw_label, label) in zip(axes, images, records):
        ax.imshow(denorm(image))
        ax.set_title(f"{path.parent.name}\nlabel={label}", fontsize=8)
        ax.axis("off")
    fig.suptitle(title)
    plt.show()

    return pd.DataFrame(
        [
            {
                "path": str(path),
                "file_name": path.name,
                "folder_name": path.parent.name,
                "raw_label": raw_label,
                "label": label,
            }
            for path, raw_label, label in records
        ]
    )


def load_vaegan_model(cfg, device):
    """Load the configured VAE-GAN checkpoint and return its path and modules."""
    from models.vaegan.loader import get_checkpoint_path, load_model

    checkpoint_path = get_checkpoint_path(cfg.model, cfg.data)
    encoder, decoder, discriminator = load_model(cfg, device)
    return checkpoint_path, encoder, decoder, discriminator


def find_training_artifacts(project_root, cfg, checkpoint_path=None):
    """Find available training-history and curve artifacts for a config."""
    candidates = []
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        candidates.extend(
            [
                checkpoint_path.with_name(f"{checkpoint_path.stem}_loss_history.csv"),
                checkpoint_path.parent / "loss_history.csv",
                checkpoint_path.parent / "training_curves.png",
            ]
        )

    dataset = getattr(cfg.data, "name", "")
    category = getattr(cfg.data, "category", "")
    model_name = getattr(cfg.model, "name", "vaegan")
    runs_root = Path(project_root) / "results" / "exp" / "runs"
    if runs_root.exists():
        patterns = [
            f"{dataset}_{category}_{model_name}_*",
            f"{dataset}_{category}_vaegan_*",
            f"*{dataset}*{category}*",
        ]
        run_dirs = []
        for pattern in patterns:
            run_dirs.extend(path for path in runs_root.glob(pattern) if path.is_dir())
        for run_dir in sorted(set(run_dirs), key=lambda path: path.stat().st_mtime, reverse=True):
            candidates.extend(
                [
                    run_dir / "loss_history.csv",
                    run_dir / "training_curves.png",
                    run_dir / "model_best_loss_history.csv",
                    run_dir / "model_last_loss_history.csv",
                ]
            )

    existing = []
    seen = set()
    for candidate in candidates:
        candidate = Path(candidate)
        if candidate.exists() and candidate not in seen:
            existing.append(candidate)
            seen.add(candidate)
    return existing


def load_training_history(project_root, cfg, checkpoint_path=None):
    """Load the first available training-history CSV and related artifacts."""
    artifacts = find_training_artifacts(project_root, cfg, checkpoint_path)
    history_paths = [path for path in artifacts if path.suffix.lower() == ".csv"]
    if not history_paths:
        return pd.DataFrame(), artifacts
    return pd.read_csv(history_paths[0]), artifacts


def show_training_curves(project_root, cfg, checkpoint_path=None):
    """Display training curves from saved history artifacts."""
    history, artifacts = load_training_history(project_root, cfg, checkpoint_path)
    curve_paths = [path for path in artifacts if path.name == "training_curves.png"]

    if history.empty:
        print("No training loss history found for this dataset/model yet.")
        if curve_paths:
            print(f"Saved curve image found: {curve_paths[0]}")
        return history, artifacts

    numeric_cols = [
        col
        for col in history.columns
        if col != "epoch" and pd.api.types.is_numeric_dtype(history[col])
    ]
    if not numeric_cols:
        print("Training history was found, but it has no numeric loss columns to plot.")
        display(history.tail())
        return history, artifacts

    x_col = "epoch" if "epoch" in history.columns else None
    n_cols = 2
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, max(3.5, 3 * n_rows)))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, col in zip(axes, numeric_cols):
        x_values = history[x_col] if x_col else history.index + 1
        ax.plot(x_values, history[col], marker="o", linewidth=1.6, markersize=3)
        ax.set_title(col)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.25)
    for ax in axes[len(numeric_cols) :]:
        ax.axis("off")

    fig.suptitle(f"{cfg.data.name} {cfg.data.category} training curves", y=1.02)
    fig.tight_layout()
    plt.show()

    print(f"Loaded training history: {artifacts[0]}")
    display(history.tail())
    return history, artifacts


def show_reconstructions(encoder, decoder, loader, device, max_images=6):
    """Display input and reconstructed images from a loader batch."""
    encoder.eval()
    decoder.eval()
    images, labels = next(iter(loader))
    images = images[:max_images].to(device)
    labels = labels[:max_images]

    with torch.inference_mode():
        mu, _ = encoder(images)
        recon = decoder(mu)

    fig, axes = plt.subplots(2, len(images), figsize=(2.4 * len(images), 4.8))
    if len(images) == 1:
        axes = axes.reshape(2, 1)
    for idx in range(len(images)):
        axes[0, idx].imshow(denorm(images[idx]))
        axes[0, idx].set_title(f"input {int(labels[idx])}", fontsize=8)
        axes[0, idx].axis("off")
        axes[1, idx].imshow(denorm(recon[idx]))
        axes[1, idx].set_title("recon", fontsize=8)
        axes[1, idx].axis("off")
    plt.show()


def run_subset_inference(
    cfg,
    encoder,
    decoder,
    discriminator,
    train_dataset,
    test_dataset,
    max_train_samples=512,
    max_test_samples=512,
):
    """Run scoring, thresholding, and metrics on train/test subsets."""
    from modules.evaluation import prepare_binary_labels, ranking_metrics, threshold_metrics
    from modules.scoring import score_samples
    from modules.thresholding import fit_threshold

    train_loader = subset_loader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        max_samples=max_train_samples,
        shuffle=False,
        num_workers=0,
    )
    test_indices = subset_indices(test_dataset, max_test_samples)
    test_loader = DataLoader(
        Subset(test_dataset, test_indices),
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=0,
    )

    train_scores, train_labels = score_samples(train_loader, encoder, decoder, discriminator, cfg)
    test_scores, test_labels = score_samples(test_loader, encoder, decoder, discriminator, cfg)
    threshold_model = fit_threshold(train_scores, cfg)
    predictions = threshold_model.predict(test_scores)
    binary_labels = prepare_binary_labels(test_labels, getattr(test_dataset, "class_to_idx", None))

    metrics = {}
    try:
        score_for_rank = (
            test_scores.detach().cpu().numpy()
            if test_scores.dim() == 1
            else test_scores.max(dim=1).values.detach().cpu().numpy()
        )
        metrics.update(ranking_metrics(score_for_rank, binary_labels))
    except Exception as exc:
        metrics["ranking_note"] = str(exc)

    metrics.update(threshold_metrics(predictions.detach().cpu().numpy(), binary_labels))

    base_records = []
    source_samples = [test_dataset.samples[idx] for idx in test_indices[: len(predictions)]]
    idx_to_class = {idx: name for name, idx in getattr(test_dataset, "class_to_idx", {}).items()}
    for (path, raw_label), binary_label, prediction, score in zip(
        source_samples,
        binary_labels,
        predictions.detach().cpu().numpy(),
        test_scores.detach().cpu().numpy(),
    ):
        path = Path(path)
        base_records.append(
            {
                "path": str(path),
                "file_name": path.name,
                "folder_name": path.parent.name,
                "class_idx": int(raw_label),
                "class_name": idx_to_class.get(int(raw_label), path.parent.name),
                "binary_label": int(binary_label),
                "prediction": int(prediction),
                "score": score.tolist() if hasattr(score, "tolist") else float(score),
            }
        )

    return metrics, pd.DataFrame(base_records), threshold_model


def save_notebook_outputs(project_root, dataset_name, category, metrics, results_df, summary):
    """Save notebook verification metrics, results, and dataset summary."""
    output_dir = project_root / "results" / "notebooks" / f"{dataset_name}_{category}_verification"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / "inference_results.csv", index=False)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with open(output_dir / "dataset_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return output_dir
