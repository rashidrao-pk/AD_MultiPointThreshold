"""
Run comparable train -> threshold calibration -> inference/evaluation sweeps.

The first registered model is ADVIS-compatible VAE-GAN, exposed as:
    advis_vaegan, vaegan, vae_gan, vae-gan, advis
"""

import argparse
import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

from data import load_data
from data.metadata import dataset_summary, sample_records
from models import get_checkpoint_loader, get_trainer, has_pretrained_loader, normalize_model_name
from modules.evaluation import prepare_binary_labels, ranking_metrics, threshold_metrics
from modules.scoring import score_samples
from modules.thresholding import fit_threshold
from utils import read_config, resolve_device, save_config_yaml


def _repo_root():
    return Path(__file__).resolve().parents[2]


def _as_list(value, default):
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _namespace_to_dict(obj):
    if hasattr(obj, "__dict__"):
        return {key: _namespace_to_dict(value) for key, value in vars(obj).items()}
    if isinstance(obj, list):
        return [_namespace_to_dict(value) for value in obj]
    return obj


def _dict_to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _dict_to_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_dict_to_namespace(item) for item in value]
    return value


def _clone_config(config):
    return _dict_to_namespace(copy.deepcopy(_namespace_to_dict(config)))


def _resolve_project_paths(config, project_root):
    data_root = Path(config.data.dataset_root)
    if not data_root.is_absolute():
        config.data.dataset_root = str(project_root / data_root)

    checkpoint_root = Path(config.model.checkpoint_root)
    if not checkpoint_root.is_absolute():
        config.model.checkpoint_root = str(project_root / checkpoint_root)

    output_dir = Path(config.output.dir)
    if not output_dir.is_absolute():
        config.output.dir = str(project_root / output_dir)


def _get_experiment_models(config):
    experiment = getattr(config, "experiment", None)
    return _as_list(getattr(experiment, "models", None), [getattr(config.model, "name", "advis_vaegan")])


def _get_scoring_setups(config):
    experiment = getattr(config, "experiment", None)
    setups = getattr(experiment, "scoring_setups", None)
    if setups:
        return setups
    return [SimpleNamespace(name=config.scoring.method, method=config.scoring.method)]


def _get_threshold_setups(config):
    experiment = getattr(config, "experiment", None)
    setups = getattr(experiment, "threshold_setups", None)
    if setups:
        return setups
    return [
        SimpleNamespace(
            name=f"{config.threshold.method}_p{config.threshold.percentile}",
            method=config.threshold.method,
            percentile=config.threshold.percentile,
            decision_rule=getattr(config.threshold, "decision_rule", "any"),
        )
    ]


def _run_dir(config, suffix=""):
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")
    suffix_part = f"_{suffix}" if suffix else ""
    return Path(config.output.dir) / "pipeline" / (
        f"{config.data.name}_{getattr(config.data, 'category', 'all')}_{timestamp}{suffix_part}"
    )


def _threshold_to_dict(threshold_model):
    out = {}
    for name in ("threshold", "thresholds", "quantiles", "decision_rule"):
        if hasattr(threshold_model, name):
            value = getattr(threshold_model, name)
            if torch.is_tensor(value):
                value = value.detach().cpu().tolist()
            out[name] = value
    return out


def _ranking_scores(scores):
    if scores.dim() == 1:
        return scores.detach().cpu().numpy()
    return scores.max(dim=1).values.detach().cpu().numpy()


def parse_args():
    parser = argparse.ArgumentParser(description="Run model/scoring/threshold/inference sweeps.")
    parser.add_argument("--config", default="configs/mvtec.yaml")
    parser.add_argument("--suffix", default="")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--skip_train", action="store_true", help="Use configured pretrained checkpoints.")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = _repo_root()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    base_config = read_config(config_path)
    _resolve_project_paths(base_config, project_root)
    if args.device is not None:
        base_config.device = args.device
    device = resolve_device(getattr(base_config, "device", "auto"))
    base_config.device = str(device)
    if args.epochs is not None:
        if not hasattr(base_config, "training"):
            base_config.training = SimpleNamespace()
        base_config.training.epochs = args.epochs

    train_loader, _, train_dataset, _ = load_data(base_config)

    eval_config = _clone_config(base_config)
    eval_config.data.augmentation = "none"
    train_score_loader, test_loader, train_score_dataset, test_dataset = load_data(eval_config)
    data_summary = {
        "train": dataset_summary(train_score_dataset, "train"),
        "test": dataset_summary(test_dataset, "test"),
    }
    run_dir = _run_dir(base_config, args.suffix)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config_yaml(base_config, run_dir / "config.yaml")
    with open(run_dir / "dataset_summary.json", "w", encoding="utf-8") as handle:
        json.dump(data_summary, handle, indent=2)

    model_names = [normalize_model_name(name) for name in _get_experiment_models(base_config)]
    scoring_setups = _get_scoring_setups(base_config)
    threshold_setups = _get_threshold_setups(base_config)

    print(f"[run] {run_dir}")
    print(f"[device] {device}")
    print(f"[data] train={len(train_dataset)} test={len(test_dataset)}")
    print(f"[data] train augmentation={getattr(base_config.data, 'augmentation', 'none')}")
    print("[data] calibration/inference augmentation=none")
    print(f"[data] train classes={data_summary['train']['class_counts']}")
    print(f"[data] test classes={data_summary['test']['class_counts']}")
    print(f"[models] {model_names}")
    print(f"[scoring] {[setup.name for setup in scoring_setups]}")
    print(f"[thresholds] {[setup.name for setup in threshold_setups]}")

    if args.dry_run:
        return

    all_results = []
    for model_name in model_names:
        model_config = _clone_config(base_config)
        model_config.model.name = model_name
        model_dir = run_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        if args.skip_train:
            if not has_pretrained_loader(model_name):
                print(f"[skip] {model_name}: no registered pretrained checkpoint loader")
                continue
            loader = get_checkpoint_loader(model_name)
            encoder, decoder, discriminator = loader(model_config, device)
            train_result_dir = None
        else:
            trainer = get_trainer(model_name)
            train_result = trainer(
                config=model_config,
                train_loader=train_loader,
                val_loader=test_loader,
                train_dataset=train_dataset,
                val_dataset=test_dataset,
                run_dir=model_dir / "training",
                device=device,
            )
            encoder = train_result["encoder"]
            decoder = train_result["decoder"]
            discriminator = train_result["discriminator"]
            train_result_dir = str(train_result["run_dir"])

        for scoring_setup in scoring_setups:
            score_config = _clone_config(model_config)
            score_config.scoring.method = scoring_setup.method
            if hasattr(scoring_setup, "quantiles"):
                score_config.scoring.quantiles = scoring_setup.quantiles

            train_scores, train_labels = score_samples(
                train_score_loader, encoder, decoder, discriminator, score_config
            )
            test_scores, test_labels = score_samples(
                test_loader, encoder, decoder, discriminator, score_config
            )
            binary_test_labels = prepare_binary_labels(
                test_labels, getattr(test_dataset, "class_to_idx", None)
            )
            test_records = sample_records(test_dataset)

            for threshold_setup in threshold_setups:
                threshold_config = _clone_config(score_config)
                threshold_config.threshold.method = threshold_setup.method
                threshold_config.threshold.percentile = threshold_setup.percentile
                threshold_config.threshold.decision_rule = getattr(threshold_setup, "decision_rule", "any")

                if threshold_config.threshold.method == "multi_point" and train_scores.dim() != 2:
                    print(
                        f"[skip] {model_name}/{scoring_setup.name}/{threshold_setup.name}: "
                        "multi_point threshold requires vector scores"
                    )
                    continue
                if threshold_config.threshold.method == "single_point" and train_scores.dim() != 1:
                    print(
                        f"[skip] {model_name}/{scoring_setup.name}/{threshold_setup.name}: "
                        "single_point threshold requires scalar scores"
                    )
                    continue

                threshold_model = fit_threshold(train_scores, threshold_config)
                predictions = threshold_model.predict(test_scores)

                rank_metrics = ranking_metrics(_ranking_scores(test_scores), binary_test_labels)
                thr_metrics = threshold_metrics(predictions.detach().cpu().numpy(), binary_test_labels)

                result = {
                    "model": model_name,
                    "scoring": scoring_setup.name,
                    "scoring_method": scoring_setup.method,
                    "threshold_setup": threshold_setup.name,
                    "threshold_method": threshold_setup.method,
                    "threshold_percentile": threshold_setup.percentile,
                    "decision_rule": getattr(threshold_setup, "decision_rule", "any"),
                    "train_dir": train_result_dir,
                    **rank_metrics,
                    **thr_metrics,
                }
                all_results.append(result)

                detail_dir = model_dir / "inference" / scoring_setup.name / threshold_setup.name
                detail_dir.mkdir(parents=True, exist_ok=True)
                result_rows = pd.DataFrame(test_records)
                result_rows["raw_label"] = test_labels.detach().cpu().numpy()
                result_rows["binary_label"] = binary_test_labels
                result_rows["prediction"] = predictions.detach().cpu().numpy()
                result_rows["score"] = (
                    test_scores.detach().cpu().numpy().tolist()
                    if test_scores.dim() > 1
                    else test_scores.detach().cpu().numpy()
                )
                result_rows.to_csv(detail_dir / "inference_results.csv", index=False)
                with open(detail_dir / "threshold.json", "w", encoding="utf-8") as handle:
                    json.dump(_threshold_to_dict(threshold_model), handle, indent=2)
                with open(detail_dir / "metrics.json", "w", encoding="utf-8") as handle:
                    json.dump(result, handle, indent=2)
                with open(detail_dir / "dataset_summary.json", "w", encoding="utf-8") as handle:
                    json.dump(data_summary, handle, indent=2)

                print(
                    f"[result] {model_name} | {scoring_setup.name} | {threshold_setup.name} "
                    f"AUROC={result.get('auroc', float('nan')):.4f} F1={result['f1']:.4f}"
                )

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(run_dir / "summary.csv", index=False)
    print(f"[done] summary={run_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
