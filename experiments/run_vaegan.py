import argparse

from utils import read_config, set_device
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


def main(config_path, suffix, force):
    """Run a full VAE-GAN anomaly detection experiment from a config file."""
    config = read_config(config_path)

    should_skip, existing = should_skip_experiment(
        config=config,
        suffix=suffix,
        force=force,
    )

    if should_skip:
        print(f"[+] Experiment already exists: {existing['id']}")
        print(f"[+] Existing run directory: {existing['run_dir']}")
        print("[+] Skipping execution. Use --force to run again.")
        return

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

    try:
        ranking_result = ranking_metrics(
            test_scores.detach().cpu(),
            binary_test_labels,
        )
    except Exception as error:
        print(f"[!] Ranking metrics failed with raw scores: {error}")
        print("[+] Ranking metrics computed using binary predictions instead.")

        ranking_result = ranking_metrics(
            prediction.detach().cpu(),
            binary_test_labels,
        )

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


def cli():
    """Parse command-line arguments and run the VAE-GAN experiment."""
    parser = argparse.ArgumentParser(description="Run anomaly detection experiment.")

    parser.add_argument("--config", required=True)
    parser.add_argument("--suffix", default="")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force execution even if the same experiment already exists.",
    )

    args = parser.parse_args()
    main(args.config, args.suffix, args.force)


if __name__ == "__main__":
    cli()
