import argparse
import numpy as np

from utils import apply_config_overrides, read_config, set_device
from data import load_data
from models.vaegan import load_model
from modules.scoring import score_samples
from modules.thresholding import fit_threshold
from modules.evaluation import (
    ranking_metrics,
    threshold_metrics,
    prepare_binary_labels,
)
from utils.experiment_saver import (
    save_experiment_run,
    should_skip_experiment,
)

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


def normalize_dataset_name(dataset):
    """Return the canonical dataset name used by configs and data loaders."""
    if dataset is None:
        return None

    key = str(dataset).strip()
    return DATASET_ALIASES.get(key.lower(), key)


def parse_category_list(value):
    """Parse comma-separated category input from the CLI."""
    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        return list(value)

    categories = [item.strip() for item in str(value).split(",") if item.strip()]
    return categories or None


def get_dataset_categories(dataset):
    """Return known categories/areas for a supported dataset."""
    dataset = normalize_dataset_name(dataset)
    if dataset not in DATASET_CATEGORIES:
        raise ValueError(
            f"Unknown dataset for category sweep: {dataset}. "
            f"Supported: {', '.join(DATASET_CATEGORIES)}"
        )
    return DATASET_CATEGORIES[dataset]


def run_single_experiment(config_path, suffix, force, overrides=None):
    """Run a full VAE-GAN anomaly detection experiment from a config file."""
    config = read_config(config_path)
    applied_overrides = apply_config_overrides(config, overrides)
    for key, value in applied_overrides:
        print(f"[+] Config override: {key} = {value}")

    should_skip, existing = should_skip_experiment(
        config=config,
        suffix=suffix,
        force=force,
    )

    if should_skip:
        print(f"[+] Experiment already exists: {existing['id']}")
        print(f"[+] Existing run directory: {existing['run_dir']}")
        print("[+] Skipping execution. Use --force to run again.")
        return {"status": "skipped", "existing": existing}

    device = set_device(config)

    train_loader, test_loader, train_dataset, test_dataset = load_data(config)

    print(f"[+] Train dataset size: {len(train_dataset)}")
    print(f"[+] Test dataset size: {len(test_dataset)}")

    enc, dec, disc = load_model(config, device)

    print(f"[+] Model loaded successfully - ({config.model.name})")
    print(f"[+] Scoring using method: {config.scoring.method}")

    train_scores, _ = score_samples(train_loader, enc, dec, disc, config)
    test_scores, test_labels = score_samples(test_loader, enc, dec, disc, config)

    print(
        f"[+] Scoring completed. "
        f"Train scores shape: {train_scores.shape}, "
        f"Test scores shape: {test_scores.shape}"
    )

    threshold_model = fit_threshold(train_scores, config)
    prediction = threshold_model.predict(test_scores)

    print("[+] Thresholding completed")

    raw_test_labels = test_labels
    binary_test_labels = prepare_binary_labels(
        test_labels,
        getattr(test_dataset, "class_to_idx", None),
    )

    results = {}

    ## importrant point here 
    # if multipoint, sahpe is not anymore (N,) but is (N, n_quantiles) so we need aggregation method to get scores
    # using mean for now, but need to find better method 
    ranking_scores = test_scores

    if test_scores.ndim == 2:
        ranking_scores = test_scores.mean(dim=1)

    ranking_result = ranking_metrics(
        ranking_scores.detach().cpu(),
        binary_test_labels,
    )
    if len(np.unique(np.asarray(binary_test_labels).astype(int))) < 2:
        print("[!] Ranking metrics are undefined because the test set has only one class.")

    results.update(ranking_result)
    print(f"[+] Ranking metrics: {ranking_result}")

    threshold_result = threshold_metrics(prediction, binary_test_labels)
    results.update(threshold_result)

    print(f"[+] Threshold metrics: {threshold_result}")

    exp_id, run_dir = save_experiment_run(
        config=config,
        results=results,
        train_scores=train_scores,
        test_scores=test_scores,
        raw_test_labels=raw_test_labels,
        binary_test_labels=binary_test_labels,
        prediction=prediction,
        test_dataset=test_dataset,
        threshold_model=threshold_model,
        suffix=suffix,
    )

    print(f"[+] Experiment {exp_id} completed.")
    print(f"[+] Results saved to: {run_dir}")
    return {"status": "done", "id": exp_id, "run_dir": str(run_dir)}


def run_experiment(
    config_path,
    suffix="",
    force=False,
    overrides=None,
    dataset=None,
    categories=None,
):
    """Run one experiment or sweep all categories for a dataset."""
    overrides = list(overrides or [])
    dataset = normalize_dataset_name(dataset)
    categories = parse_category_list(categories)

    if categories == ["all"]:
        if dataset is None:
            base_config = read_config(config_path)
            dataset = normalize_dataset_name(base_config.data.name)
        categories = get_dataset_categories(dataset)

    if dataset is None and not categories:
        return [run_single_experiment(config_path, suffix, force, overrides)]

    if dataset is None:
        base_config = read_config(config_path)
        dataset = normalize_dataset_name(base_config.data.name)

    if not categories:
        categories = get_dataset_categories(dataset)

    results = []
    total = len(categories)
    for index, category in enumerate(categories, start=1):
        print("=" * 80)
        print(f"[+] Dataset sweep {index}/{total}: {dataset}/{category}")
        print("=" * 80)
        run_overrides = [
            *overrides,
            f"data.name={dataset}",
            f"data.category={category}",
        ]
        results.append(
            run_single_experiment(
                config_path=config_path,
                suffix=suffix,
                force=force,
                overrides=run_overrides,
            )
        )

    done = sum(1 for item in results if item and item.get("status") == "done")
    skipped = sum(1 for item in results if item and item.get("status") == "skipped")
    print("=" * 80)
    print(f"[+] Sweep completed: done={done}, skipped={skipped}, total={total}")
    print("=" * 80)
    return results


def main(config_path, suffix, force, overrides=None):
    """Backward-compatible single-experiment entry point."""
    return run_single_experiment(config_path, suffix, force, overrides)


def cli():
    """Parse command-line arguments and run the VAE-GAN experiment."""
    parser = argparse.ArgumentParser(description="Run anomaly detection experiment.")

    parser.add_argument("--config", required=True)
    parser.add_argument("--suffix", default="")
    parser.add_argument(
        "--category",
        default=None,
        help=(
            "Override data.category, for example --category zipper. "
            "Use --category all with --dataset to sweep all known categories."
        ),
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "Override data.name and run all known categories for that dataset "
            "unless --category selects one category. Examples: MVTec, Cobots_Synthetic."
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
        "--device",
        default=None,
        help="Override device, for example auto, cpu, cuda, or mps.",
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
            "--set threshold.percentile=99."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force execution even if the same experiment already exists.",
    )

    args = parser.parse_args()
    overrides = list(args.set_overrides)
    sweep_categories = None
    if args.category is not None and args.category.lower() == "all":
        sweep_categories = ["all"]
    elif args.category is not None:
        if args.dataset is not None:
            sweep_categories = [args.category]
        else:
            overrides.append(f"data.category={args.category}")
    if args.dataset_root is not None:
        overrides.append(f"data.dataset_root={args.dataset_root}")
    if args.checkpoint_root is not None:
        overrides.append(f"model.checkpoint_root={args.checkpoint_root}")
    if args.device is not None:
        overrides.append(f"device={args.device}")

    run_experiment(
        config_path=args.config,
        suffix=args.suffix,
        force=args.force,
        overrides=overrides,
        dataset=args.dataset,
        categories=sweep_categories,
    )


if __name__ == "__main__":
    cli()
