import csv
import json
import time
from pathlib import Path

import torch


def get_next_experiment_id(experiments_dir: Path, width: int = 4) -> str:
    experiments_dir.mkdir(parents=True, exist_ok=True)

    existing_ids = []
    for path in experiments_dir.iterdir():
        if path.is_dir() and path.name.startswith("E"):
            try:
                existing_ids.append(int(path.name[1:]))
            except ValueError:
                pass

    next_id = max(existing_ids, default=0) + 1
    return f"E{next_id:0{width}d}"


def flatten_metrics(metrics: dict) -> dict:
    flat = {}

    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()

        if hasattr(value, "item"):
            value = value.item()

        flat[key] = value

    return flat


def append_runs_csv(csv_path: Path, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))

        if not file_exists:
            writer.writeheader()

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

    exp_id = get_next_experiment_id(experiments_dir)
    run_dir = experiments_dir / exp_id
    run_dir.mkdir(parents=True, exist_ok=False)

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    outputs_path = run_dir / "outputs.pt"
    results_path = run_dir / "results.json"
    config_path = run_dir / "config.yaml"

    torch.save(
        {
            "train_scores": train_scores.detach().cpu(),
            "test_scores": test_scores.detach().cpu(),
            "raw_test_labels": raw_test_labels.detach().cpu(),
            "binary_test_labels": binary_test_labels,
            "prediction": prediction.detach().cpu(),
        },
        outputs_path,
    )

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(flatten_metrics(results), f, indent=4)

    from utils import save_config_yaml
    save_config_yaml(config, config_path)

    row = {
        "id": exp_id,
        "date": timestamp,
        "model": config.model.name,
        "dataset": config.data.name,
        "category": getattr(config.data, "category", ""),
        "scoring": config.scoring.method,
        "threshold": config.threshold.method,
        "decision_rule": getattr(config.threshold, "decision_rule", ""),
        "suffix": suffix,
        "status": "done",
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "results_path": str(results_path),
        "outputs_path": str(outputs_path),
    }

    row.update(flatten_metrics(results))
    append_runs_csv(runs_csv_path, row)

    return exp_id, run_dir