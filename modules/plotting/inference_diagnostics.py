"""Reusable inference diagnostic plots for anomaly-detection experiment runs."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from .training_plots import save_latent_space


NORMAL_CLASS_NAMES = {"good", "normal", "ok", "healthy"}


def _to_image(tensor):
    """Convert a normalized CHW tensor into an HWC image array."""
    return ((tensor.detach().cpu().clamp(-1, 1) + 1) / 2).permute(1, 2, 0).numpy()


def _anomaly_map(input_image, reconstructed_image):
    """Return a single-channel absolute reconstruction-error map."""
    return torch.mean(torch.abs(input_image - reconstructed_image), dim=0).detach().cpu().numpy()


def _kde_values(scores, x_values):
    """Return KDE density values for scores when enough samples are available."""
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if len(scores) < 2 or np.allclose(scores, scores[0]):
        return None

    try:
        from scipy.stats import gaussian_kde

        return gaussian_kde(scores)(x_values)
    except Exception:
        return None


def _score_axis_values(*score_groups):
    """Return a shared x-axis span for score distribution plots."""
    values = np.concatenate([scores for scores in score_groups if len(scores)])
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    span = max(max_value - min_value, 1e-6)
    padding = span * 0.08
    return np.linspace(max(0.0, min_value - padding), max_value + padding, 256)


def _class_names_from_loader(loader):
    """Return class-index to class-name mapping from a dataloader dataset."""
    dataset = getattr(loader, "dataset", None)
    classes = getattr(dataset, "classes", None)
    if classes:
        return {idx: name for idx, name in enumerate(classes)}
    class_to_idx = getattr(dataset, "class_to_idx", None)
    if class_to_idx:
        return {idx: name for name, idx in class_to_idx.items()}
    return {}


def _is_normal_label(label, class_names):
    """Return whether a numeric label should be treated as normal."""
    label_name = class_names.get(int(label), "")
    if str(label_name).strip().lower() in NORMAL_CLASS_NAMES:
        return True
    return int(label) == 0 and not label_name


def save_inference_score_distribution(df, output_path):
    """Save binary normal/anomalous score histograms and KDE curves for inference."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    normal_scores = df[df["true_label"] == 0]["score"].to_numpy(dtype=float)
    anomaly_scores = df[df["true_label"] == 1]["score"].to_numpy(dtype=float)
    if len(normal_scores) == 0 and len(anomaly_scores) == 0:
        return None

    x_values = _score_axis_values(normal_scores, anomaly_scores)
    bins = np.linspace(float(x_values[0]), float(x_values[-1]), 40)
    threshold = None
    if "threshold" in df.columns and df["threshold"].notna().any():
        threshold = float(df["threshold"].dropna().iloc[0])

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.2), constrained_layout=True)
    for ax, mode in zip(axes, ("hist", "kde")):
        for scores, label, color in (
            (normal_scores, "Normal", "forestgreen"),
            (anomaly_scores, "Anomalous", "darkorange"),
        ):
            if len(scores) == 0:
                continue
            if mode == "hist":
                ax.hist(scores, bins=bins, density=True, alpha=0.30, color=color, label=label)
                ax.hist(
                    scores, bins=bins, density=True, histtype="step", linewidth=1.3, color=color
                )
                ax.axvline(float(np.median(scores)), color=color, linestyle="--", linewidth=1.2)
            else:
                density = _kde_values(scores, x_values)
                if density is not None:
                    ax.plot(x_values, density, color=color, linewidth=1.8, label=label)
                    ax.fill_between(x_values, 0, density, color=color, alpha=0.18)
                    ax.axvline(float(np.mean(scores)), color=color, linestyle="--", linewidth=1.2)
                    ax.axvline(float(np.median(scores)), color=color, linestyle=":", linewidth=1.2)

        if threshold is not None:
            ax.axvline(threshold, color="black", linestyle="-.", linewidth=1.5, label="Threshold")
        ax.set_xlabel("Anomaly score")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    axes[0].set_title("Histogram of inference scores")
    axes[1].set_title("KDE of inference scores")
    fig.suptitle("Normal vs anomalous inference score distribution", fontsize=11)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def save_inference_latent_space(
    encoder, loader, device, output_path, max_batches=8, projection="pca"
):
    """Save latent-space projection for an inference/test dataloader."""
    return save_latent_space(
        encoder=encoder,
        loader=loader,
        device=device,
        output_path=output_path,
        max_batches=max_batches,
        default_label="test samples",
        projection=projection,
    )


def _select_validation_examples(encoder, decoder, loader, device, max_batches=8):
    """Select one normal and one anomalous sample with reconstructions."""
    encoder.eval()
    decoder.eval()
    class_names = _class_names_from_loader(loader)
    selected = {}

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches or {"normal", "anomalous"} <= set(selected):
                break

            images, labels = batch[0].to(device), batch[1]
            mu, _ = encoder(images)
            recon = decoder(mu)
            labels = torch.as_tensor(labels).detach().cpu().numpy()

            for idx, label in enumerate(labels):
                key = "normal" if _is_normal_label(label, class_names) else "anomalous"
                if key in selected:
                    continue
                selected[key] = {
                    "image": images[idx].detach().cpu(),
                    "reconstruction": recon[idx].detach().cpu(),
                    "label": class_names.get(int(label), str(label)),
                    "score": float(torch.mean(torch.abs(images[idx] - recon[idx])).detach().cpu()),
                }

    return selected


def save_inference_validation_samples(
    encoder,
    decoder,
    loader,
    device,
    output_path,
    max_batches=8,
):
    """Save normal/anomalous inference samples with reconstruction and anomaly map."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected = _select_validation_examples(
        encoder=encoder,
        decoder=decoder,
        loader=loader,
        device=device,
        max_batches=max_batches,
    )
    if not selected:
        return None

    rows = [
        (name, item)
        for name, item in (
            ("normal", selected.get("normal")),
            ("anomalous", selected.get("anomalous")),
        )
        if item
    ]
    fig, axes = plt.subplots(len(rows), 3, figsize=(8.8, 3.0 * len(rows)), constrained_layout=True)
    if len(rows) == 1:
        axes = np.asarray([axes])

    for row_idx, (group_name, item) in enumerate(rows):
        image = item["image"]
        recon = item["reconstruction"]
        panels = (
            ("Input", _to_image(image), None),
            ("Reconstruction", _to_image(recon), None),
            (f"Anomaly map\nscore={item['score']:.5f}", _anomaly_map(image, recon), "magma"),
        )
        axes[row_idx, 0].text(
            -0.08,
            0.5,
            f"{group_name.title()}\n{item['label']}",
            transform=axes[row_idx, 0].transAxes,
            rotation=90,
            ha="right",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
        for col_idx, (title, panel, cmap) in enumerate(panels):
            axes[row_idx, col_idx].imshow(panel, cmap=cmap)
            axes[row_idx, col_idx].set_title(title, fontsize=9)
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])

    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path
