"""Training-quality diagnostic plots for anomaly-detection models."""

from pathlib import Path
import csv

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


NORMAL_CLASS_NAMES = {"good", "normal", "ok", "healthy"}


def _to_image(tensor):
    """Convert a normalized CHW tensor into an HWC image array."""
    return ((tensor.detach().cpu().clamp(-1, 1) + 1) / 2).permute(1, 2, 0).numpy()


def _anomaly_map(input_image, reconstructed_image):
    """Return a single-channel absolute reconstruction-error map."""
    return torch.mean(torch.abs(input_image - reconstructed_image), dim=0).detach().cpu().numpy()


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
    """Return whether a label represents a normal class."""
    label_name = class_names.get(int(label), "")
    if str(label_name).strip().lower() in NORMAL_CLASS_NAMES:
        return True
    return int(label) == 0 and not label_name


def _safe_mean(values):
    """Return a finite mean or NaN for an empty value array."""
    values = np.asarray(values, dtype=float)
    return float(np.mean(values)) if len(values) else float("nan")


def _safe_percentile(values, percentile):
    """Return a percentile or NaN for an empty value array."""
    values = np.asarray(values, dtype=float)
    return float(np.percentile(values, percentile)) if len(values) else float("nan")


def _tpr_at_fpr(y_true, scores, target_fpr):
    """Return true-positive rate at or below a requested false-positive rate."""
    try:
        from sklearn.metrics import roc_curve

        fpr, tpr, _ = roc_curve(y_true, scores)
    except Exception:
        return float("nan")

    valid = tpr[fpr <= target_fpr]
    return float(np.max(valid)) if len(valid) else 0.0


def _binary_metrics(y_true, scores):
    """Return AUROC, AUPRC, and low-FPR recall metrics for binary labels."""
    if len(set(np.asarray(y_true).tolist())) < 2:
        return {
            "auroc": float("nan"),
            "auprc": float("nan"),
            "tpr_at_1_fpr": float("nan"),
            "tpr_at_5_fpr": float("nan"),
        }

    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        auroc = float(roc_auc_score(y_true, scores))
        auprc = float(average_precision_score(y_true, scores))
    except Exception:
        auroc = float("nan")
        auprc = float("nan")

    return {
        "auroc": auroc,
        "auprc": auprc,
        "tpr_at_1_fpr": _tpr_at_fpr(y_true, scores, 0.01),
        "tpr_at_5_fpr": _tpr_at_fpr(y_true, scores, 0.05),
    }


def _append_csv_row(path, row):
    """Append a row to a CSV file while preserving existing columns."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        with open(path, "r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            old_rows = list(reader)
            fieldnames = list(dict.fromkeys((reader.fieldnames or []) + list(row.keys())))
    else:
        old_rows = []
        fieldnames = list(row.keys())

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(old_rows)
        writer.writerow(row)


def _collect_loader_outputs(encoder, decoder, discriminator, loader, device, max_batches=8):
    """Collect scores, labels, component scores, and latent radii from a dataloader."""
    encoder.eval()
    decoder.eval()
    if discriminator is not None:
        discriminator.eval()

    records = []
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break

            images = batch[0].to(device)
            labels = batch[1] if len(batch) > 1 else torch.zeros(len(images), dtype=torch.long)
            labels = torch.as_tensor(labels).detach().cpu().numpy()
            mu, _ = encoder(images)
            recon = decoder(mu)
            abs_diff = torch.abs(images - recon)
            l1_scores = torch.mean(abs_diff, dim=(1, 2, 3))
            l2_scores = torch.sqrt(torch.mean((images - recon) ** 2, dim=(1, 2, 3)))
            mse_scores = torch.mean((images - recon) ** 2, dim=(1, 2, 3))
            max_scores = torch.amax(torch.mean(abs_diff, dim=1), dim=(1, 2))
            latent_radius = torch.sum(mu ** 2, dim=1)
            if discriminator is None:
                dis_scores = torch.full_like(l1_scores, float("nan"))
            else:
                dis_scores = torch.sigmoid(discriminator(recon)).flatten()

            for idx in range(len(images)):
                records.append(
                    {
                        "label": int(labels[idx]),
                        "score_l1": float(l1_scores[idx].detach().cpu()),
                        "score_l2": float(l2_scores[idx].detach().cpu()),
                        "score_mse": float(mse_scores[idx].detach().cpu()),
                        "score_max": float(max_scores[idx].detach().cpu()),
                        "discriminator_score": float(dis_scores[idx].detach().cpu()),
                        "latent_radius": float(latent_radius[idx].detach().cpu()),
                        "image": images[idx].detach().cpu(),
                        "reconstruction": recon[idx].detach().cpu(),
                    }
                )

    return records


def _records_to_arrays(records, class_names):
    """Convert collected records into numpy arrays for metrics and plotting."""
    labels = np.asarray([record["label"] for record in records], dtype=int)
    is_anomaly = np.asarray(
        [not _is_normal_label(label, class_names) for label in labels],
        dtype=bool,
    )
    return {
        "labels": labels,
        "is_anomaly": is_anomaly,
        "score_l1": np.asarray([record["score_l1"] for record in records], dtype=float),
        "score_l2": np.asarray([record["score_l2"] for record in records], dtype=float),
        "score_mse": np.asarray([record["score_mse"] for record in records], dtype=float),
        "score_max": np.asarray([record["score_max"] for record in records], dtype=float),
        "discriminator_score": np.asarray(
            [record["discriminator_score"] for record in records],
            dtype=float,
        ),
        "latent_radius": np.asarray([record["latent_radius"] for record in records], dtype=float),
    }


def _dataset_sample_path(dataset, index):
    """Return a dataset sample path when the dataset exposes one."""
    samples = getattr(dataset, "samples", None) or getattr(dataset, "imgs", None)
    if samples and index < len(samples):
        sample = samples[index]
        if isinstance(sample, (list, tuple)) and sample:
            return str(sample[0])
    return ""


def _validation_grid_indices(dataset, class_names):
    """Select one normal and one anomalous validation sample by class name."""
    targets = getattr(dataset, "targets", None)
    if targets is None:
        samples = getattr(dataset, "samples", None) or []
        targets = [
            sample[1]
            for sample in samples
            if isinstance(sample, (list, tuple)) and len(sample) > 1
        ]

    selected = {"normal": None, "anomalous": None}
    for index, label in enumerate(targets):
        key = "normal" if _is_normal_label(label, class_names) else "anomalous"
        if selected[key] is None:
            selected[key] = index
        if selected["normal"] is not None and selected["anomalous"] is not None:
            break

    return selected


def _collect_validation_grid_records(encoder, decoder, discriminator, loader, device):
    """Collect deterministic normal/anomalous examples for the validation grid."""
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        return []

    class_names = _class_names_from_loader(loader)
    selected = _validation_grid_indices(dataset, class_names)
    indices = [
        index
        for index in (selected["normal"], selected["anomalous"])
        if index is not None
    ]
    if not indices:
        return []

    encoder.eval()
    decoder.eval()
    if discriminator is not None:
        discriminator.eval()

    records = []
    with torch.inference_mode():
        for index in indices:
            image, label = dataset[index]
            image = image.detach().cpu()
            batch = image.unsqueeze(0).to(device)
            mu, _ = encoder(batch)
            recon = decoder(mu)
            abs_diff = torch.abs(batch - recon)
            l1_score = torch.mean(abs_diff, dim=(1, 2, 3))[0]
            l2_score = torch.sqrt(torch.mean((batch - recon) ** 2, dim=(1, 2, 3)))[0]
            mse_score = torch.mean((batch - recon) ** 2, dim=(1, 2, 3))[0]
            max_score = torch.amax(torch.mean(abs_diff, dim=1), dim=(1, 2))[0]
            latent_radius = torch.sum(mu ** 2, dim=1)[0]
            if discriminator is None:
                dis_score = torch.tensor(float("nan"))
            else:
                dis_score = torch.sigmoid(discriminator(recon)).flatten()[0]

            records.append(
                {
                    "label": int(label),
                    "score_l1": float(l1_score.detach().cpu()),
                    "score_l2": float(l2_score.detach().cpu()),
                    "score_mse": float(mse_score.detach().cpu()),
                    "score_max": float(max_score.detach().cpu()),
                    "discriminator_score": float(dis_score.detach().cpu()),
                    "latent_radius": float(latent_radius.detach().cpu()),
                    "image": image,
                    "reconstruction": recon.squeeze(0).detach().cpu(),
                    "path": _dataset_sample_path(dataset, index),
                }
            )

    return records


def collect_quality_snapshot(
    encoder,
    decoder,
    discriminator,
    train_loader,
    val_loader,
    device,
    max_batches=8,
):
    """Collect one epoch of anomaly-detection quality measurements."""
    class_names = _class_names_from_loader(val_loader)
    train_records = _collect_loader_outputs(
        encoder,
        decoder,
        discriminator,
        train_loader,
        device,
        max_batches=max_batches,
    )
    val_records = _collect_loader_outputs(
        encoder,
        decoder,
        discriminator,
        val_loader,
        device,
        max_batches=max_batches,
    )
    train_arrays = _records_to_arrays(train_records, _class_names_from_loader(train_loader))
    val_arrays = _records_to_arrays(val_records, class_names)
    y_true = val_arrays["is_anomaly"].astype(int)
    scores = val_arrays["score_l1"]
    normal_scores = scores[~val_arrays["is_anomaly"]]
    if len(normal_scores) == 0:
        normal_scores = train_arrays["score_l1"]
    anomaly_scores = scores[val_arrays["is_anomaly"]]
    metrics = {
        **_binary_metrics(y_true, scores),
        "normal_mean_score": _safe_mean(normal_scores),
        "anomaly_mean_score": _safe_mean(anomaly_scores),
        "score_gap": _safe_mean(anomaly_scores) - _safe_mean(normal_scores),
        "threshold_p95": _safe_percentile(train_arrays["score_l1"], 95),
        "threshold_p99": _safe_percentile(train_arrays["score_l1"], 99),
    }
    return {
        "metrics": metrics,
        "train": train_arrays,
        "val": val_arrays,
        "val_records": val_records,
        "validation_grid_records": _collect_validation_grid_records(
            encoder,
            decoder,
            discriminator,
            val_loader,
            device,
        ),
        "class_names": class_names,
    }


def save_quality_metrics_history(metrics_history, output_path):
    """Save epoch-wise AUROC, AUPRC, score-gap, and threshold diagnostics."""
    if not metrics_history:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history = pd.DataFrame(metrics_history)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), constrained_layout=True)
    panels = (
        (axes[0, 0], [("normal_mean_score", "Normal mean"), ("anomaly_mean_score", "Anomalous mean"), ("score_gap", "Gap")], "Score separation"),
        (axes[0, 1], [("auroc", "AUROC"), ("auprc", "AUPRC")], "Ranking metrics"),
        (axes[1, 0], [("tpr_at_1_fpr", "TPR @ 1% FPR"), ("tpr_at_5_fpr", "TPR @ 5% FPR")], "Recall at fixed false-positive rate"),
        (axes[1, 1], [("threshold_p95", "Normal P95"), ("threshold_p99", "Normal P99")], "Threshold stability"),
    )
    for ax, columns, title in panels:
        for column, label in columns:
            if column in history:
                ax.plot(history["epoch"], history[column], marker="o", linewidth=1.4, label=label)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_loss_balance(loss_history, output_path):
    """Save weighted loss contributions over epochs."""
    if not loss_history:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history = pd.DataFrame(loss_history)
    columns = [
        ("recon_loss", "Recon"),
        ("beta_kl_loss", "Beta KL"),
        ("beta_gan_loss", "Beta GAN"),
        ("beta_center_loss", "Beta center"),
        ("beta_svdd_loss", "Beta SVDD"),
        ("disc_loss", "Disc"),
    ]

    fig, ax = plt.subplots(figsize=(10.6, 5.8), constrained_layout=True)
    for column, label in columns:
        if column in history and history[column].notna().any():
            ax.plot(history["epoch"], history[column], linewidth=1.5, label=label)
    ax.set_title("Weighted loss balance")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss contribution")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_score_components(snapshot, output_path):
    """Save normal/anomalous distributions for multiple anomaly-score components."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    val = snapshot["val"]
    is_anomaly = val["is_anomaly"]
    components = [
        ("score_l1", "L1"),
        ("score_l2", "L2"),
        ("score_mse", "MSE"),
        ("score_max", "Max pixel"),
        ("discriminator_score", "Discriminator"),
    ]

    fig, axes = plt.subplots(1, len(components), figsize=(3.0 * len(components), 4.8), constrained_layout=True)
    for ax, (key, title) in zip(axes, components):
        normal = val[key][~is_anomaly]
        anomaly = val[key][is_anomaly]
        data = [normal[np.isfinite(normal)], anomaly[np.isfinite(anomaly)]]
        ax.boxplot(data, labels=["Normal", "Anomalous"], showfliers=False, patch_artist=True)
        ax.scatter(np.full(len(data[0]), 1), data[0], s=9, alpha=0.35, color="forestgreen")
        ax.scatter(np.full(len(data[1]), 2), data[1], s=9, alpha=0.35, color="darkorange")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle("Score components by validation class")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_latent_radius_distribution(snapshot, output_path):
    """Save normal/anomalous latent-radius distributions."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    val = snapshot["val"]
    normal = val["latent_radius"][~val["is_anomaly"]]
    anomaly = val["latent_radius"][val["is_anomaly"]]
    values = np.concatenate([normal, anomaly]) if len(normal) or len(anomaly) else np.array([0.0])
    bins = np.linspace(float(np.min(values)), float(np.max(values) + 1e-6), 34)

    fig, ax = plt.subplots(figsize=(8.6, 5.3), constrained_layout=True)
    if len(normal):
        ax.hist(normal, bins=bins, density=True, alpha=0.30, color="forestgreen", label="Normal")
    if len(anomaly):
        ax.hist(anomaly, bins=bins, density=True, alpha=0.30, color="darkorange", label="Anomalous")
    ax.axvline(_safe_mean(normal), color="forestgreen", linestyle="--", linewidth=1.2)
    ax.axvline(_safe_mean(anomaly), color="darkorange", linestyle="--", linewidth=1.2)
    ax.set_title("Latent radius distribution")
    ax.set_xlabel("Squared distance from latent center")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _first_record(records, class_names, anomalous):
    """Return the first normal or anomalous record from a record list."""
    for record in records:
        is_anomaly = not _is_normal_label(record["label"], class_names)
        if is_anomaly == anomalous:
            return record
    return None


def save_validation_quality_grid(snapshot, output_path):
    """Save fixed validation examples with reconstruction, map, score, and label."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = snapshot.get("validation_grid_records") or snapshot["val_records"]
    class_names = snapshot["class_names"]
    selected = [
        ("Normal", _first_record(records, class_names, anomalous=False)),
        ("Anomalous", _first_record(records, class_names, anomalous=True)),
    ]
    selected = [(name, record) for name, record in selected if record is not None]
    if not selected:
        return None

    fig, axes = plt.subplots(len(selected), 3, figsize=(8.8, 3.0 * len(selected)), constrained_layout=True)
    if len(selected) == 1:
        axes = np.asarray([axes])

    for row_idx, (group_name, record) in enumerate(selected):
        image = record["image"]
        recon = record["reconstruction"]
        label_name = class_names.get(int(record["label"]), str(record["label"]))
        sample_path = str(record.get("path", ""))
        path_hint = f"\n{Path(sample_path).parent.name}/{Path(sample_path).name}" if sample_path else ""
        score = record["score_l1"]
        panels = (
            ("Input", _to_image(image), None),
            ("Reconstruction", _to_image(recon), None),
            (f"Anomaly map\nscore={score:.4f}", _anomaly_map(image, recon), "magma"),
        )
        axes[row_idx, 0].text(
            -0.08,
            0.5,
            f"{group_name}\n{label_name}{path_hint}",
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

    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_quality_diagnostics(
    epoch,
    loss_history,
    encoder,
    decoder,
    discriminator,
    train_loader,
    val_loader,
    device,
    run_dir,
    max_batches=8,
):
    """Collect and save all training-quality diagnostic plots for one epoch."""
    run_dir = Path(run_dir)
    snapshot = collect_quality_snapshot(
        encoder,
        decoder,
        discriminator,
        train_loader,
        val_loader,
        device,
        max_batches=max_batches,
    )
    row = {"epoch": epoch, **snapshot["metrics"]}
    history_path = run_dir / "quality_metrics.csv"
    _append_csv_row(history_path, row)
    metrics_history = pd.read_csv(history_path).to_dict("records")

    save_quality_metrics_history(
        metrics_history,
        run_dir / "plots" / "quality_metrics" / f"quality_metrics_epoch_{epoch:04d}.png",
    )
    save_score_components(
        snapshot,
        run_dir / "plots" / "score_components" / f"score_components_epoch_{epoch:04d}.png",
    )
    save_latent_radius_distribution(
        snapshot,
        run_dir / "plots" / "latent_radius" / f"latent_radius_epoch_{epoch:04d}.png",
    )
    save_loss_balance(
        loss_history,
        run_dir / "plots" / "loss_balance" / f"loss_balance_epoch_{epoch:04d}.png",
    )
    save_validation_quality_grid(
        snapshot,
        run_dir / "plots" / "validation_quality" / f"validation_quality_epoch_{epoch:04d}.png",
    )
    return snapshot
