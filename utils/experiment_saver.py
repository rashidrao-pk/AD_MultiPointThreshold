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


def save_experiment_run(
    config,
    results: dict,
    train_scores,
    test_scores,
    raw_test_labels,
    binary_test_labels,
    prediction,
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
    results_path = run_dir / "results.json"
    config_path = run_dir / "config.yaml"

    clean_results = flatten_metrics(results)

    torch.save(
        {
            "train_scores": detach_cpu(train_scores),
            "test_scores": detach_cpu(test_scores),
            "raw_test_labels": detach_cpu(raw_test_labels),
            "binary_test_labels": detach_cpu(binary_test_labels),
            "prediction": detach_cpu(prediction),
        },
        outputs_path,
    )

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(clean_results, f, indent=4)

    save_config_yaml(config, config_path)

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
    }

    row.update(clean_results)
    append_runs_csv(runs_csv_path, row)

    return exp_id, run_dir