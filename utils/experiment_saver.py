import csv
import hashlib
import json
import time
from pathlib import Path

import torch
import yaml


def config_to_dict(obj):
    if isinstance(obj, dict):
        return {key: config_to_dict(value) for key, value in obj.items()}

    if hasattr(obj, "__dict__"):
        return {key: config_to_dict(value) for key, value in vars(obj).items()}

    if isinstance(obj, (list, tuple)):
        return [config_to_dict(value) for value in obj]

    return obj


def experiment_hash(config, suffix: str = "") -> str:
    signature = {
        "data": config_to_dict(config.data),
        "model": config_to_dict(config.model),
        "scoring": config_to_dict(config.scoring),
        "threshold": config_to_dict(config.threshold),
        "suffix": suffix,
    }

    payload = json.dumps(signature, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def find_existing_experiment(runs_csv_path: Path, exp_hash: str):
    if not runs_csv_path.exists():
        return None

    with open(runs_csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if "hash" not in (reader.fieldnames or []):
            return None

        for row in reader:
            if row.get("hash") == exp_hash and row.get("status") == "done":
                return row

    return None


def should_skip_experiment(config, suffix: str = "", force: bool = False):
    if force:
        return False, None

    runs_csv_path = Path(config.output.dir) / "runs.csv"
    exp_hash = experiment_hash(config, suffix)
    existing = find_existing_experiment(runs_csv_path, exp_hash)

    return existing is not None, existing


def get_next_experiment_id_from_csv(runs_csv_path: Path, width: int = 5) -> str:
    if not runs_csv_path.exists():
        return f"E{1:0{width}d}"

    existing_ids = []

    with open(runs_csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if "id" not in (reader.fieldnames or []):
            return f"E{1:0{width}d}"

        for row in reader:
            exp_id = row.get("id", "")

            if exp_id.startswith("E"):
                try:
                    existing_ids.append(int(exp_id[1:]))
                except ValueError:
                    continue

    return f"E{max(existing_ids, default=0) + 1:0{width}d}"


def to_python(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()

        if value.numel() == 1:
            return value.item()

        return value.tolist()

    if hasattr(value, "item"):
        return value.item()

    return value


def flatten_metrics(metrics: dict) -> dict:
    return {key: to_python(value) for key, value in metrics.items()}


def detach_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()

    return value


def save_config_yaml(config, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_to_dict(config), f, sort_keys=False)


def append_runs_csv(csv_path: Path, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists():
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            old_fieldnames = reader.fieldnames or []
            old_rows = list(reader)

        fieldnames = list(dict.fromkeys(old_fieldnames + list(row.keys())))
    else:
        old_rows = []
        fieldnames = list(row.keys())

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(old_rows)
        writer.writerow(row)


def tensor_to_list(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    return list(x)


def get_threshold_payload(threshold_model):
    payload = {}

    if hasattr(threshold_model, "threshold"):
        payload["threshold"] = to_python(threshold_model.threshold)

    if hasattr(threshold_model, "thresholds"):
        payload["thresholds"] = to_python(threshold_model.thresholds)

    if hasattr(threshold_model, "quantiles"):
        payload["quantiles"] = to_python(threshold_model.quantiles)

    if hasattr(threshold_model, "decision_rule"):
        payload["decision_rule"] = threshold_model.decision_rule

    return payload


def classify_error(true_label, pred_label):
    true_label = int(true_label)
    pred_label = int(pred_label)

    if true_label == 1 and pred_label == 1:
        return "TP"
    if true_label == 0 and pred_label == 0:
        return "TN"
    if true_label == 0 and pred_label == 1:
        return "FP"
    if true_label == 1 and pred_label == 0:
        return "FN"

    return "UNKNOWN"


def get_dataset_paths_and_names(dataset):
    """
    Supports:
    - torchvision.datasets.ImageFolder
    - custom datasets with .samples = [(path, label), ...]
    """
    paths = []
    class_names = []

    samples = getattr(dataset, "samples", None)
    classes = getattr(dataset, "classes", None)

    if samples is None:
        return paths, class_names

    for item in samples:
        path, raw_label = item[0], item[1]
        paths.append(str(path))

        if classes is not None and int(raw_label) < len(classes):
            class_names.append(classes[int(raw_label)])
        else:
            class_names.append(str(raw_label))

    return paths, class_names


def score_to_columns(score):
    """
    Converts scalar scores or vector scores to CSV-friendly columns.

    Single-point:
        score = 0.123

    Multi-point:
        score_q0 = ...
        score_q1 = ...
        score_mean = ...
        score_max = ...
    """
    if isinstance(score, torch.Tensor):
        score = score.detach().cpu()

    if not torch.is_tensor(score):
        score = torch.tensor(score)

    score = score.float()

    if score.numel() == 1:
        return {
            "score": float(score.item()),
            "score_mean": float(score.item()),
            "score_max": float(score.item()),
        }

    values = score.view(-1).tolist()

    row = {
        "score": float(max(values)),
        "score_mean": float(sum(values) / len(values)),
        "score_max": float(max(values)),
    }

    for i, value in enumerate(values):
        row[f"score_q{i}"] = float(value)

    return row


def save_predictions_csv_json(
    csv_path: Path,
    json_path: Path,
    config,
    test_dataset,
    test_scores,
    raw_test_labels,
    binary_test_labels,
    prediction,
    threshold_model,
):
    paths, class_names = get_dataset_paths_and_names(test_dataset)

    test_scores = detach_cpu(test_scores)
    raw_test_labels = tensor_to_list(raw_test_labels)
    binary_test_labels = tensor_to_list(binary_test_labels)
    prediction = tensor_to_list(prediction)

    threshold_payload = get_threshold_payload(threshold_model)

    rows = []

    n = len(binary_test_labels)

    for i in range(n):
        row = {
            "index": i,
            "image_path": paths[i] if i < len(paths) else "",
            "class_name": class_names[i] if i < len(class_names) else "",
            "raw_label": int(raw_test_labels[i]),
            "true_label": int(binary_test_labels[i]),
            "pred_label": int(prediction[i]),
            "error_type": classify_error(binary_test_labels[i], prediction[i]),
            "dataset": getattr(config.data, "name", ""),
            "category": getattr(config.data, "category", ""),
            "model": getattr(config.model, "name", ""),
            "latent_dims": getattr(config.model, "latent_dims", ""),
            "score_method": getattr(config.scoring, "method", ""),
            "threshold_method": getattr(config.threshold, "method", ""),
            "threshold_percentile": getattr(config.threshold, "percentile", ""),
            "decision_rule": getattr(config.threshold, "decision_rule", ""),
        }

        if "threshold" in threshold_payload:
            row["threshold"] = threshold_payload["threshold"]

        if "thresholds" in threshold_payload:
            for j, value in enumerate(threshold_payload["thresholds"]):
                row[f"threshold_q{j}"] = value

        score_row = score_to_columns(test_scores[i])
        row.update(score_row)

        rows.append(row)

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if rows:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=4)

    return rows


def save_experiment_run(
    config,
    results: dict,
    train_scores,
    test_scores,
    raw_test_labels,
    binary_test_labels,
    prediction,
    test_dataset=None,
    threshold_model=None,
    suffix: str = "",
):
    root = Path(config.output.dir)
    experiments_dir = root / "experiments"
    runs_csv_path = root / "runs.csv"

    exp_hash = experiment_hash(config, suffix)
    exp_id = get_next_experiment_id_from_csv(runs_csv_path)

    run_dir = experiments_dir / exp_id
    run_dir.mkdir(parents=True, exist_ok=False)

    outputs_path = run_dir / "outputs.pt"
    results_path = run_dir / "metrics.json"
    config_path = run_dir / "config.yaml"
    predictions_csv_path = run_dir / "predictions.csv"
    predictions_json_path = run_dir / "predictions.json"

    clean_results = flatten_metrics(results)

    threshold_payload = {}
    if threshold_model is not None:
        threshold_payload = get_threshold_payload(threshold_model)

    torch.save(
        {
            "train_scores": detach_cpu(train_scores),
            "test_scores": detach_cpu(test_scores),
            "raw_test_labels": detach_cpu(raw_test_labels),
            "binary_test_labels": detach_cpu(binary_test_labels),
            "prediction": detach_cpu(prediction),
            "threshold": threshold_payload,
        },
        outputs_path,
    )

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(clean_results, f, indent=4)

    save_config_yaml(config, config_path)

    if test_dataset is not None and threshold_model is not None:
        save_predictions_csv_json(
            csv_path=predictions_csv_path,
            json_path=predictions_json_path,
            config=config,
            test_dataset=test_dataset,
            test_scores=test_scores,
            raw_test_labels=raw_test_labels,
            binary_test_labels=binary_test_labels,
            prediction=prediction,
            threshold_model=threshold_model,
        )

    row = {
        "id": exp_id,
        "hash": exp_hash,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "done",
        "model": config.model.name,
        "dataset": config.data.name,
        "category": getattr(config.data, "category", ""),
        "scoring": config.scoring.method,
        "threshold": config.threshold.method,
        "percentile": getattr(config.threshold, "percentile", ""),
        "decision_rule": getattr(config.threshold, "decision_rule", ""),
        "suffix": suffix,
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "results_path": str(results_path),
        "outputs_path": str(outputs_path),
        "predictions_csv": str(predictions_csv_path),
        "predictions_json": str(predictions_json_path),
    }

    row.update(clean_results)
    append_runs_csv(runs_csv_path, row)

    return exp_id, run_dir