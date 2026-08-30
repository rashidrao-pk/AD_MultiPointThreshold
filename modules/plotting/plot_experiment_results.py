import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay

from data import load_data
from models.vaegan import load_model
from utils import read_config
from utils.experiment_saver import experiment_hash
from .inference_diagnostics import (
    save_inference_latent_space,
    save_inference_score_distribution,
    save_inference_validation_samples,
)


def _resolve_project_path(path):
    """Resolve a path relative to the project root."""
    path = Path(str(path).replace("\\", "/"))
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _read_runs_csv(path):
    """Read experiment registry rows from a runs.csv file."""
    if not path.exists():
        return []

    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _row_matches_config(row, config, suffix=""):
    """Return whether a runs.csv row matches the selected experiment config."""
    if row.get("hash") == experiment_hash(config, suffix):
        return True

    return all(
        str(row.get(column, "")) == str(value)
        for column, value in (
            ("status", "done"),
            ("model", getattr(config.model, "name", "")),
            ("dataset", getattr(config.data, "name", "")),
            ("category", getattr(config.data, "category", "")),
            ("scoring", getattr(config.scoring, "method", "")),
            ("threshold", getattr(config.threshold, "method", "")),
            ("percentile", getattr(config.threshold, "percentile", "")),
            ("decision_rule", getattr(config.threshold, "decision_rule", "")),
            ("suffix", suffix),
        )
    )


def resolve_run_dir_from_config(config_path, suffix=""):
    """Find the latest completed experiment run directory matching a config file."""
    config = read_config(config_path)
    output_dir = _resolve_project_path(config.output.dir)
    runs_csv_path = output_dir / "runs.csv"
    rows = [
        row
        for row in _read_runs_csv(runs_csv_path)
        if row.get("status") == "done" and _row_matches_config(row, config, suffix=suffix)
    ]

    if not rows:
        raise FileNotFoundError(
            f"No completed experiment found for config {config_path} in {runs_csv_path}."
        )

    row = rows[-1]
    run_dir = row.get("run_dir") or row.get("path")
    if not run_dir:
        raise ValueError(f"Matched runs.csv row has no run_dir: {row}")

    return _resolve_project_path(run_dir)


def read_predictions(run_dir):
    """Load per-sample predictions from an experiment run directory."""
    run_dir = Path(run_dir)
    pred_path = run_dir / "predictions.csv"

    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions.csv: {pred_path}")

    return pd.read_csv(pred_path)


def save_confusion_matrix(df, out_dir):
    """Save a confusion-matrix plot for binary predictions."""
    fig, ax = plt.subplots(figsize=(5, 5))

    ConfusionMatrixDisplay.from_predictions(
        df["true_label"],
        df["pred_label"],
        display_labels=["normal", "anomaly"],
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )

    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=300)
    plt.close(fig)


def save_roc_pr(df, out_dir):
    """Save ROC and precision-recall curve plots when both classes are present."""
    if df["true_label"].nunique() < 2:
        print("[!] Skipping ROC/PR: only one class is present.")
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(
        df["true_label"],
        df["score"],
        ax=ax,
    )
    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    PrecisionRecallDisplay.from_predictions(
        df["true_label"],
        df["score"],
        ax=ax,
    )
    ax.set_title("Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(out_dir / "precision_recall_curve.png", dpi=300)
    plt.close(fig)


def save_score_distribution(df, out_dir):
    """Save normal/anomaly score distribution histograms."""
    return save_inference_score_distribution(df, out_dir / "score_distribution.png")


def save_error_type_bar(df, out_dir):
    """Save a bar chart of TN, FP, FN, and TP counts."""
    counts = df["error_type"].value_counts().reindex(["TN", "FP", "FN", "TP"]).fillna(0)

    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax)

    ax.set_title("Prediction Outcome Counts")
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "outcome_counts.png", dpi=300)
    plt.close(fig)


def safe_open_image(path):
    """Open an image path as RGB, returning None on failure."""
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def select_top_cases(df, error_type, k=5):
    """Select the most informative rows for a prediction outcome type."""
    sub = df[df["error_type"] == error_type].copy()

    if len(sub) == 0:
        return sub

    if error_type == "FP":
        return sub.sort_values("score", ascending=False).head(k)

    if error_type == "FN":
        return sub.sort_values("score", ascending=True).head(k)

    if error_type == "TP":
        return sub.sort_values("score", ascending=False).head(k)

    if error_type == "TN":
        return sub.sort_values("score", ascending=True).head(k)

    return sub.head(k)


def save_case_gallery(df, out_dir, error_type, k=5):
    """Save an image gallery and CSV for selected prediction outcome cases."""
    selected = select_top_cases(df, error_type, k=k)

    if len(selected) == 0:
        print(f"[!] No {error_type} samples found.")
        return

    gallery_dir = out_dir / f"top_{error_type.lower()}"
    gallery_dir.mkdir(parents=True, exist_ok=True)

    n = len(selected)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

    if n == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, selected.iterrows()):
        img = safe_open_image(row["image_path"])

        if img is not None:
            ax.imshow(img)

        ax.set_title(
            f"{error_type}\nscore={row['score']:.5f}\n{Path(row['image_path']).parent.name}",
            fontsize=9,
        )
        ax.axis("off")

    fig.suptitle(f"Top {n} {error_type} cases")
    fig.tight_layout()
    fig.savefig(out_dir / f"top_{error_type.lower()}_gallery.png", dpi=300)
    plt.close(fig)

    selected.to_csv(gallery_dir / f"top_{error_type.lower()}.csv", index=False)


def save_per_class_performance(df, out_dir):
    """Save per-class metrics and an optional mean-score plot."""
    if "class_name" not in df.columns:
        return

    rows = []

    for class_name, g in df.groupby("class_name"):
        tp = ((g["true_label"] == 1) & (g["pred_label"] == 1)).sum()
        fp = ((g["true_label"] == 0) & (g["pred_label"] == 1)).sum()
        fn = ((g["true_label"] == 1) & (g["pred_label"] == 0)).sum()
        tn = ((g["true_label"] == 0) & (g["pred_label"] == 0)).sum()

        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)

        rows.append(
            {
                "class_name": class_name,
                "count": len(g),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "mean_score": float(g["score"].mean()),
            }
        )

    perf = pd.DataFrame(rows)
    perf.to_csv(out_dir / "per_class_performance.csv", index=False)

    if len(perf) > 1:
        perf_sorted = perf.sort_values("mean_score", ascending=True)

        fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(perf_sorted))))
        ax.barh(perf_sorted["class_name"], perf_sorted["mean_score"])
        ax.set_title("Mean anomaly score by class")
        ax.set_xlabel("Mean anomaly score")
        ax.grid(axis="x", alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_dir / "mean_score_by_class.png", dpi=300)
        plt.close(fig)


def save_model_backed_diagnostics(run_dir, out_dir):
    """Save latent-space and validation reconstruction plots when config/model are available."""
    config_path = Path(run_dir) / "config.yaml"
    if not config_path.exists():
        print(f"[!] Skipping model-backed diagnostics: missing {config_path}")
        return

    try:
        config = read_config(config_path)
        config.device = "cpu"
        device = torch.device("cpu")
        _, test_loader, _, _ = load_data(config)
        encoder, decoder, _ = load_model(config, device)
    except Exception as error:
        print(f"[!] Skipping model-backed diagnostics: {error}")
        return

    save_inference_latent_space(
        encoder,
        test_loader,
        device,
        out_dir / "latent_space.png",
        max_batches=12,
        projection="pca",
    )
    save_inference_validation_samples(
        encoder,
        decoder,
        test_loader,
        device,
        out_dir / "validation_samples.png",
        max_batches=12,
    )


def main(run_dir):
    """Generate all result plots for an experiment run directory."""
    run_dir = Path(run_dir)
    out_dir = run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_predictions(run_dir)

    save_confusion_matrix(df, out_dir)
    save_roc_pr(df, out_dir)
    save_score_distribution(df, out_dir)
    save_error_type_bar(df, out_dir)

    for error_type in ["FP", "FN", "TP", "TN"]:
        save_case_gallery(df, out_dir, error_type, k=5)

    save_per_class_performance(df, out_dir)
    save_model_backed_diagnostics(run_dir, out_dir)

    print(f"[+] Saved plots to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir", default=None, help="Experiment run directory, e.g. results/experiments/E00008."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Config file used to find the latest matching experiment run.",
    )
    parser.add_argument(
        "--suffix", default="", help="Optional experiment suffix used when matching --config."
    )
    args = parser.parse_args()

    if args.run_dir is None and args.config is None:
        parser.error("one of --run_dir or --config is required")

    selected_run_dir = args.run_dir
    if selected_run_dir is None:
        selected_run_dir = resolve_run_dir_from_config(args.config, suffix=args.suffix)
        print(f"[+] Selected experiment run: {selected_run_dir}")

    main(selected_run_dir)
