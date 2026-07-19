"""Plotting helpers for model training runs."""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError
import torch

from .training_quality_plots import save_quality_diagnostics


def _to_image(tensor):
    """Convert a normalized CHW tensor into an HWC image array."""
    return ((tensor.detach().cpu().clamp(-1, 1) + 1) / 2).permute(1, 2, 0).numpy()


def save_training_curves(loss_history, output_path):
    """Save training and validation loss curves to an image file."""
    if not loss_history:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history = pd.DataFrame(loss_history)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    model_loss_column = "ae_loss" if "ae_loss" in history else "vae_loss"
    if model_loss_column == "ae_loss":
        model_loss_title = "Autoencoder loss"
    elif "gan_loss" in history:
        model_loss_title = "VAE-GAN generator loss"
    else:
        model_loss_title = "VAE loss"
    curve_specs = [
        (axes[0, 0], "recon_loss", "Reconstruction loss"),
        (axes[0, 1], model_loss_column, model_loss_title),
        (axes[1, 0], "disc_loss", "Discriminator loss"),
        (axes[1, 1], "val_recon_loss", "Validation reconstruction loss"),
    ]

    for ax, column, title in curve_specs:
        if column in history:
            ax.plot(history["epoch"], history[column], linewidth=1.3)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.25)
        else:
            ax.set_axis_off()

    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_reconstruction_preview(encoder, decoder, loader, device, output_path, max_images=6):
    """Save a side-by-side preview of input and reconstructed images."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    encoder.eval()
    decoder.eval()
    images, _ = next(iter(loader))
    images = images[:max_images].to(device)

    with torch.inference_mode():
        mu, _ = encoder(images)
        recon = decoder(mu)

    fig, axes = plt.subplots(
        2, len(images), figsize=(2.4 * len(images), 4.8), constrained_layout=True
    )
    if len(images) == 1:
        axes = axes.reshape(2, 1)
    for idx in range(len(images)):
        axes[0, idx].imshow(_to_image(images[idx]))
        axes[0, idx].set_title("input", fontsize=8)
        axes[0, idx].axis("off")
        axes[1, idx].imshow(_to_image(recon[idx]))
        axes[1, idx].set_title("recon", fontsize=8)
        axes[1, idx].axis("off")

    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _anomaly_map(input_image, reconstructed_image):
    """Return a single-channel absolute reconstruction-error map."""
    return torch.mean(torch.abs(input_image - reconstructed_image), dim=0).detach().cpu().numpy()


def _reconstruction_scores(input_images, reconstructed_images):
    """Return one reconstruction-error anomaly score per image."""
    return torch.mean(torch.abs(input_images - reconstructed_images), dim=(1, 2, 3))


def _decode_latent_interpolation(
    encoder, decoder, discriminator, start_image, end_image, device, steps=6
):
    """Decode evenly spaced latent vectors between two encoded images."""
    start_batch = start_image.unsqueeze(0).to(device)
    end_batch = end_image.unsqueeze(0).to(device)

    with torch.inference_mode():
        start_mu, _ = encoder(start_batch)
        end_mu, _ = encoder(end_batch)
        weights = torch.linspace(0.0, 1.0, steps, device=device).view(-1, 1)
        latents = (1.0 - weights) * start_mu + weights * end_mu
        generated = decoder(latents)
        scores = None
        if discriminator is not None:
            scores = torch.sigmoid(discriminator(generated)).detach().cpu().flatten().numpy()

    return generated.detach().cpu(), scores


def _show_image_panel(ax, image, title, cmap=None):
    """Render one image-style panel without ticks."""
    ax.imshow(image, cmap=cmap)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def _add_group_border(fig, axes, label):
    """Draw a solid figure-level border around a group of axes."""
    fig.canvas.draw()
    bboxes = [ax.get_position() for ax in axes]
    left = min(box.x0 for box in bboxes)
    bottom = min(box.y0 for box in bboxes)
    right = max(box.x1 for box in bboxes)
    top = max(box.y1 for box in bboxes)
    pad = 0.006

    border = Rectangle(
        (left - pad, bottom - pad),
        (right - left) + 2 * pad,
        (top - bottom) + 2 * pad,
        transform=fig.transFigure,
        fill=False,
        linewidth=2.2,
        edgecolor="black",
        zorder=20,
    )
    fig.add_artist(border)
    fig.text(
        left + 0.006,
        top - 0.006,
        label,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.5},
        zorder=30,
    )


def save_training_evolution(
    encoder,
    decoder,
    discriminator,
    fixed_image,
    random_image,
    device,
    output_path,
    anomaly_examples=None,
    interpolation_steps=6,
):
    """Save normal/anomalous reconstruction and latent-interpolation diagnostics."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    encoder.eval()
    decoder.eval()
    if discriminator is not None:
        discriminator.eval()

    fixed_image = fixed_image.detach().cpu()
    random_image = random_image.detach().cpu()
    anomaly_examples = list(anomaly_examples or [])[:2]
    images = {
        "fixed normal": fixed_image,
        "random normal": random_image,
    }
    for index, example in enumerate(anomaly_examples, start=1):
        images[f"anomaly {index}"] = example["image"].detach().cpu()

    reconstructions = {}
    with torch.inference_mode():
        for name, image in images.items():
            batch = image.unsqueeze(0).to(device)
            mu, _ = encoder(batch)
            reconstructions[name] = decoder(mu).squeeze(0).detach().cpu()

    interpolated, interpolation_scores = _decode_latent_interpolation(
        encoder,
        decoder,
        discriminator,
        fixed_image,
        random_image,
        device,
        steps=interpolation_steps,
    )

    interpolation_cols = max(1, int(np.ceil(interpolation_steps / 2)))
    n_cols = max(3 + interpolation_cols, 6 if anomaly_examples else 0)
    n_rows = 3 if anomaly_examples else 2
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.4 * n_cols, 2.6 * n_rows),
        constrained_layout=True,
    )

    row_names = ("Fixed normal sample", "Random normal sample")
    for row_idx, name in enumerate(("fixed normal", "random normal")):
        image = images[name]
        recon = reconstructions[name]
        panels = (
            ("input", _to_image(image), None),
            ("reconstruction", _to_image(recon), None),
            ("anomaly map", _anomaly_map(image, recon), "magma"),
        )
        axes[row_idx, 0].text(
            -0.08,
            0.5,
            row_names[row_idx],
            transform=axes[row_idx, 0].transAxes,
            rotation=90,
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
        for col_idx, (title, panel, cmap) in enumerate(panels):
            _show_image_panel(axes[row_idx, col_idx], panel, title, cmap=cmap)

    if anomaly_examples:
        axes[2, 0].text(
            -0.08,
            0.5,
            "Anomalous samples",
            transform=axes[2, 0].transAxes,
            rotation=90,
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
        for example_idx, example in enumerate(anomaly_examples):
            name = f"anomaly {example_idx + 1}"
            image = images[name]
            recon = reconstructions[name]
            class_name = example.get("class_name") or "anomaly"
            start_col = example_idx * 3
            panels = (
                (f"{class_name}\ninput", _to_image(image), None),
                ("reconstruction", _to_image(recon), None),
                ("anomaly map", _anomaly_map(image, recon), "magma"),
            )
            for offset, (title, panel, cmap) in enumerate(panels):
                _show_image_panel(axes[2, start_col + offset], panel, title, cmap=cmap)

        for col_idx in range(len(anomaly_examples) * 3, n_cols):
            axes[2, col_idx].axis("off")

    dis_axes = []
    for step_idx in range(interpolation_steps):
        row_idx = step_idx // interpolation_cols
        col_idx = 3 + (step_idx % interpolation_cols)
        ax = axes[row_idx, col_idx]
        _show_image_panel(ax, _to_image(interpolated[step_idx]), f"step {step_idx + 1}")
        title = f"step {col_idx + 1}"
        if interpolation_scores is not None:
            title = f"step {step_idx + 1}\nD={interpolation_scores[step_idx]:.2f}"
        ax.set_title(title, fontsize=9)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
            spine.set_edgecolor("black")
        dis_axes.append(ax)

    for step_idx in range(interpolation_steps, 2 * interpolation_cols):
        row_idx = step_idx // interpolation_cols
        col_idx = 3 + (step_idx % interpolation_cols)
        axes[row_idx, col_idx].axis("off")

    _add_group_border(fig, dis_axes, "Discriminator latent interpolation")

    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _project_latents(latents, method="tsne"):
    """Project latent vectors to two dimensions for plotting."""
    if latents.shape[1] == 1:
        return latents[:, [0]], None

    if latents.shape[1] == 2:
        return latents, "latent dimensions"

    method = str(method).lower()
    if method == "tsne" and len(latents) >= 3:
        try:
            from sklearn.manifold import TSNE

            perplexity = min(30, max(2, (len(latents) - 1) // 3))
            projected = TSNE(
                n_components=2,
                perplexity=perplexity,
                init="pca",
                learning_rate="auto",
                random_state=42,
            ).fit_transform(latents)
            return projected, "t-SNE projection"
        except Exception:
            pass

    try:
        from sklearn.decomposition import PCA

        return PCA(n_components=2).fit_transform(latents), "PCA projection"
    except Exception:
        return latents[:, :2], "first two latent dimensions"


def _class_names_from_loader(loader):
    """Return class names from a dataloader dataset when available."""
    dataset = getattr(loader, "dataset", None)
    classes = getattr(dataset, "classes", None)
    if classes:
        return {idx: name for idx, name in enumerate(classes)}
    class_to_idx = getattr(dataset, "class_to_idx", None)
    if class_to_idx:
        return {idx: name for name, idx in class_to_idx.items()}
    return {}


def _is_normal_class(label, label_name):
    """Return whether a class label should be treated as normal."""
    normal_names = {"good", "normal", "ok", "healthy"}
    if label_name and str(label_name).strip().lower() in normal_names:
        return True
    return int(label) == 0 and not label_name


def _cluster_hull(points, color):
    """Build a translucent convex hull covering a class cluster."""
    if len(points) < 3:
        return None

    try:
        hull = ConvexHull(points)
    except QhullError:
        return None

    vertices = points[hull.vertices]
    center = vertices.mean(axis=0)
    vertices = center + 1.04 * (vertices - center)

    return Polygon(
        vertices,
        closed=True,
        facecolor=color,
        edgecolor=color,
        alpha=0.12,
        linewidth=1.5,
        zorder=1,
    )


def _axis_limits_path(output_path):
    """Return the persistent axis-limit sidecar path for latent plots."""
    return Path(output_path).parent / "latent_axis_limits.json"


def _padded_limits(values, padding_ratio=0.08):
    """Return min/max limits with a small padding."""
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    span = max(max_value - min_value, 1e-6)
    padding = span * padding_ratio
    return min_value - padding, max_value + padding


def _stable_axis_limits(output_path, x_values, y_values, projection):
    """Load and update stable latent plot axis limits for a run."""
    path = _axis_limits_path(output_path)
    current = {
        "projection": str(projection).lower(),
        "xlim": list(_padded_limits(x_values)),
        "ylim": list(_padded_limits(y_values)),
    }

    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as handle:
                saved = json.load(handle)
            if saved.get("projection") == current["projection"]:
                current["xlim"] = [
                    min(float(saved["xlim"][0]), current["xlim"][0]),
                    max(float(saved["xlim"][1]), current["xlim"][1]),
                ]
                current["ylim"] = [
                    min(float(saved["ylim"][0]), current["ylim"][0]),
                    max(float(saved["ylim"][1]), current["ylim"][1]),
                ]
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            pass

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(current, handle, indent=2)

    return current["xlim"], current["ylim"]


def save_latent_space(
    encoder,
    loader,
    device,
    output_path,
    max_batches=8,
    class_names=None,
    default_label="normal train",
    projection="tsne",
):
    """Save a 2D plot of encoded latent vectors from a dataloader."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    encoder.eval()
    latent_batches = []
    label_batches = []

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break

            images = batch[0]
            labels = batch[1] if len(batch) > 1 else None
            images = images.to(device)
            mu, _ = encoder(images)
            latent_batches.append(mu.detach().cpu())
            if labels is not None:
                label_batches.append(torch.as_tensor(labels).detach().cpu())

    if not latent_batches:
        return None

    latents = torch.cat(latent_batches).numpy()
    labels = torch.cat(label_batches).numpy() if label_batches else None
    class_names = class_names or _class_names_from_loader(loader)
    projected, projection_name = _project_latents(latents, method=projection)

    if projected.shape[1] == 1:
        x_values = projected[:, 0]
        y_values = projected[:, 0] * 0.0
        xlabel = "Latent dimension 1"
        ylabel = ""
    else:
        x_values = projected[:, 0]
        y_values = projected[:, 1]
        xlabel = "Component 1"
        ylabel = "Component 2"

    fig, ax = plt.subplots(figsize=(8.8, 6), constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if labels is None:
        color = colors[0] if colors else "C0"
        points = np.column_stack([x_values, y_values])
        hull = _cluster_hull(points, color)
        if hull is not None:
            ax.add_patch(hull)
        ax.scatter(
            x_values,
            y_values,
            s=18,
            alpha=0.82,
            marker="o",
            color=color,
            edgecolors="white",
            linewidths=0.35,
            label=default_label,
            zorder=2,
        )
    else:
        for color_idx, label in enumerate(sorted(set(labels.tolist()))):
            mask = labels == label
            label_name = class_names.get(int(label), f"class {label}")
            color = colors[color_idx % len(colors)] if colors else f"C{color_idx}"
            marker = "o" if _is_normal_class(label, label_name) else "X"
            size = 18 if marker == "o" else 28
            points = np.column_stack([x_values[mask], y_values[mask]])
            hull = _cluster_hull(points, color)
            if hull is not None:
                ax.add_patch(hull)
            ax.scatter(
                x_values[mask],
                y_values[mask],
                s=size,
                alpha=0.86,
                marker=marker,
                color=color,
                edgecolors="white",
                linewidths=0.35,
                label=label_name,
                zorder=2,
            )

    title = "Latent space"
    if projection_name:
        title = f"{title} ({projection_name})"
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    xlim, ylim = _stable_axis_limits(output_path, x_values, y_values, projection)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _collect_score_distribution(encoder, decoder, loader, device, max_batches=8):
    """Collect reconstruction-error scores and labels from a dataloader."""
    encoder.eval()
    decoder.eval()
    score_batches = []
    label_batches = []

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break

            images = batch[0].to(device)
            labels = batch[1] if len(batch) > 1 else None
            mu, _ = encoder(images)
            recon = decoder(mu)
            score_batches.append(_reconstruction_scores(images, recon).detach().cpu())
            if labels is not None:
                label_batches.append(torch.as_tensor(labels).detach().cpu())

    if not score_batches:
        return np.array([]), None

    scores = torch.cat(score_batches).numpy()
    labels = torch.cat(label_batches).numpy() if label_batches else None
    return scores, labels


def _binary_score_groups(train_scores, val_scores, val_labels, class_names):
    """Collapse score labels into normal and anomalous groups."""
    normal_parts = []
    anomaly_parts = []

    if len(train_scores):
        normal_parts.append(train_scores)

    if len(val_scores):
        if val_labels is None:
            anomaly_parts.append(val_scores)
        else:
            for label in sorted(set(val_labels.tolist())):
                mask = val_labels == label
                label_name = class_names.get(int(label), f"class {label}")
                if _is_normal_class(label, label_name):
                    normal_parts.append(val_scores[mask])
                else:
                    anomaly_parts.append(val_scores[mask])

    normal_scores = np.concatenate(normal_parts) if normal_parts else np.array([])
    anomaly_scores = np.concatenate(anomaly_parts) if anomaly_parts else np.array([])
    return normal_scores, anomaly_scores


def _kde_values(scores, x_values):
    """Return KDE density values for scores, falling back safely for tiny samples."""
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


def _plot_hist_and_kde(ax_hist, ax_kde, scores, label, color, bins, x_values):
    """Plot one score group as a histogram and KDE curve."""
    if len(scores) == 0:
        return

    ax_hist.hist(
        scores,
        bins=bins,
        density=True,
        alpha=0.28,
        color=color,
        label=f"{label} histogram",
        histtype="stepfilled",
    )
    ax_hist.hist(
        scores,
        bins=bins,
        density=True,
        alpha=0.9,
        color=color,
        histtype="step",
        linewidth=1.3,
    )

    median = float(np.median(scores))
    mean = float(np.mean(scores))
    ax_hist.axvline(median, color=color, linestyle="--", linewidth=1.2, alpha=0.85)

    density = _kde_values(scores, x_values)
    if density is not None:
        ax_kde.plot(x_values, density, color=color, linewidth=1.8, label=f"{label} KDE")
        ax_kde.fill_between(x_values, 0, density, color=color, alpha=0.18)
        ax_kde.axvline(mean, color=color, linestyle="--", linewidth=1.3, alpha=0.9)
        ax_kde.axvline(median, color=color, linestyle=":", linewidth=1.2, alpha=0.9)


def save_score_distribution(
    encoder,
    decoder,
    train_loader,
    val_loader,
    device,
    output_path,
    max_batches=8,
):
    """Save binary normal/anomalous score histograms and KDE curves."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    train_scores, _ = _collect_score_distribution(
        encoder,
        decoder,
        train_loader,
        device,
        max_batches=max_batches,
    )
    val_scores, val_labels = _collect_score_distribution(
        encoder,
        decoder,
        val_loader,
        device,
        max_batches=max_batches,
    )

    if len(train_scores) == 0 and len(val_scores) == 0:
        return None

    class_names = _class_names_from_loader(val_loader)
    normal_scores, anomaly_scores = _binary_score_groups(
        train_scores,
        val_scores,
        val_labels,
        class_names,
    )
    if len(normal_scores) == 0 and len(anomaly_scores) == 0:
        return None

    x_values = _score_axis_values(normal_scores, anomaly_scores)
    bins = np.linspace(float(x_values[0]), float(x_values[-1]), 34)
    normal_color = "forestgreen"
    anomaly_color = "darkorange"

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.2), constrained_layout=True)
    ax_hist, ax_kde = axes
    _plot_hist_and_kde(
        ax_hist,
        ax_kde,
        normal_scores,
        "Normal",
        normal_color,
        bins,
        x_values,
    )
    _plot_hist_and_kde(
        ax_hist,
        ax_kde,
        anomaly_scores,
        "Anomalous",
        anomaly_color,
        bins,
        x_values,
    )

    normal_mean = float(np.mean(normal_scores)) if len(normal_scores) else None
    anomaly_mean = float(np.mean(anomaly_scores)) if len(anomaly_scores) else None
    title_bits = ["Score distribution"]
    if normal_mean is not None:
        title_bits.append(f"Normal mean: {normal_mean:.4f}")
    if anomaly_mean is not None:
        title_bits.append(f"Anomalous mean: {anomaly_mean:.4f}")

    ax_hist.set_title("Histogram of anomaly scores")
    ax_hist.set_xlabel("Mean absolute reconstruction error")
    ax_hist.set_ylabel("Density")
    ax_hist.grid(True, alpha=0.25)
    ax_hist.legend(loc="upper right")

    ax_kde.set_title("KDE of normal and anomalous scores")
    ax_kde.set_xlabel("Mean absolute reconstruction error")
    ax_kde.set_ylabel("Density")
    ax_kde.grid(True, alpha=0.25)
    ax_kde.legend(loc="upper right")
    ax_kde.text(
        0.98,
        0.05,
        "\n".join(title_bits[1:]),
        transform=ax_kde.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
    )
    fig.suptitle(" - ".join(title_bits), fontsize=11)

    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


class TrainingPlotter:
    """Coordinate optional plots during a training run."""

    def __init__(
        self,
        run_dir,
        plot_curves=False,
        plot_latent_space=False,
        plot_score_distribution=False,
        plot_quality=False,
        latent_space_classes="normal",
        latent_projection="tsne",
        plot_every=10,
        max_latent_batches=8,
        max_score_batches=8,
    ):
        """Store plotting options and output locations."""
        self.run_dir = Path(run_dir)
        self.plot_curves = plot_curves
        self.plot_latent_space = plot_latent_space
        self.plot_score_distribution = plot_score_distribution
        self.plot_quality = plot_quality
        self.latent_space_classes = latent_space_classes
        self.latent_projection = latent_projection
        self.plot_every = max(int(plot_every), 1)
        self.max_latent_batches = max(int(max_latent_batches), 1)
        self.max_score_batches = max(int(max_score_batches), 1)
        self.fixed_image = None
        self.fixed_anomaly_examples = None

    def should_plot(self, epoch, total_epochs):
        """Return whether plots should be saved for this epoch."""
        return epoch == 1 or epoch % self.plot_every == 0 or epoch == total_epochs

    def _sample_normal_image(self, loader):
        """Return one image from the normal training loader."""
        images = next(iter(loader))[0].detach().cpu()
        index = torch.randint(low=0, high=len(images), size=(1,)).item()
        return images[index]

    def _class_name_for_label(self, dataset, label):
        """Return a display class name for an integer label."""
        label = int(label)
        classes = getattr(dataset, "classes", None)
        if classes is not None and 0 <= label < len(classes):
            return str(classes[label])

        class_to_idx = getattr(dataset, "class_to_idx", None) or {}
        for name, idx in class_to_idx.items():
            if int(idx) == label:
                return str(name)

        return f"class_{label}"

    def _normal_label_ids(self, dataset):
        """Return labels that should be treated as normal/non-anomalous."""
        normal_names = {"good", "normal"}
        class_to_idx = getattr(dataset, "class_to_idx", None) or {}
        normal_ids = {
            int(idx) for name, idx in class_to_idx.items() if str(name).lower() in normal_names
        }
        if not normal_ids:
            normal_ids.add(0)
        return normal_ids

    def _sample_anomaly_examples(self, loader, max_examples=2):
        """Return anomalous examples, preferring different anomaly classes."""
        dataset = getattr(loader, "dataset", None)
        normal_ids = self._normal_label_ids(dataset)
        by_label = {}

        for images, labels in loader:
            images = images.detach().cpu()
            labels = labels.detach().cpu()
            for image, label_tensor in zip(images, labels):
                label = int(label_tensor.item())
                if label in normal_ids:
                    continue
                by_label.setdefault(label, []).append(image)
            if (
                sum(len(items) for items in by_label.values()) >= max_examples
                and len(by_label) >= max_examples
            ):
                break

        examples = []
        for label in sorted(by_label):
            if by_label[label]:
                examples.append(
                    {
                        "image": by_label[label][0],
                        "label": label,
                        "class_name": self._class_name_for_label(dataset, label),
                    }
                )
            if len(examples) >= max_examples:
                return examples

        if len(examples) < max_examples:
            for label in sorted(by_label):
                for image in by_label[label][1:]:
                    examples.append(
                        {
                            "image": image,
                            "label": label,
                            "class_name": self._class_name_for_label(dataset, label),
                        }
                    )
                    if len(examples) >= max_examples:
                        return examples

        return examples

    def on_epoch_end(
        self,
        epoch,
        total_epochs,
        loss_history,
        encoder,
        decoder,
        discriminator,
        train_loader,
        val_loader,
        device,
    ):
        """Save all enabled plots for a completed epoch."""
        if not self.should_plot(epoch, total_epochs):
            return

        if self.fixed_image is None:
            self.fixed_image = self._sample_normal_image(train_loader)
        if self.fixed_anomaly_examples is None:
            self.fixed_anomaly_examples = self._sample_anomaly_examples(val_loader)
        random_image = self._sample_normal_image(train_loader)

        if self.plot_curves:
            save_training_curves(loss_history, self.run_dir / "training_curves.png")
            save_training_curves(
                loss_history,
                self.run_dir
                / "plots"
                / "training_curves"
                / f"training_curves_epoch_{epoch:04d}.png",
            )
            save_reconstruction_preview(
                encoder,
                decoder,
                train_loader,
                device,
                self.run_dir / "reconstruction_preview.png",
            )
            save_training_evolution(
                encoder,
                decoder,
                discriminator,
                self.fixed_image,
                random_image,
                device,
                self.run_dir
                / "plots"
                / "training_evolution"
                / f"training_evolution_epoch_{epoch:04d}.png",
                anomaly_examples=self.fixed_anomaly_examples,
            )

        if self.plot_latent_space:
            latent_loader = train_loader
            latent_label = "normal train"
            if self.latent_space_classes == "both":
                latent_loader = val_loader
                latent_label = "validation/test"

            save_latent_space(
                encoder,
                latent_loader,
                device,
                self.run_dir / "plots" / "latent_space" / f"latent_space_epoch_{epoch:04d}.png",
                max_batches=self.max_latent_batches,
                default_label=latent_label,
                projection=self.latent_projection,
            )

        if self.plot_score_distribution:
            save_score_distribution(
                encoder,
                decoder,
                train_loader,
                val_loader,
                device,
                self.run_dir
                / "plots"
                / "score_distribution"
                / f"score_distribution_epoch_{epoch:04d}.png",
                max_batches=self.max_score_batches,
            )

        if self.plot_quality:
            save_quality_diagnostics(
                epoch,
                loss_history,
                encoder,
                decoder,
                discriminator,
                train_loader,
                val_loader,
                device,
                self.run_dir,
                max_batches=self.max_score_batches,
            )
