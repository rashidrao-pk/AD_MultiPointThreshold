import argparse
from utils import read_config, make_run_dir, save_config_yaml
import torch
import time
import json
from pathlib import Path
import yaml

from data import load_data
from models.vaegan import load_model
from modules.scoring import score_samples
from modules.thresholding import fit_threshold
from modules.evaluation import ranking_metrics, threshold_metrics, prepare_binary_labels




def main(config_path, suffix):
    config = read_config(config_path)

    # Set up device for GPU training
    device = config.device if torch.cuda.is_available() else "cpu"
    print(f"[+] Using device: {device}")

    # loading the data

    train_loader, test_loader, train_dataset, test_dataset = load_data(config)

    print(f"[+] Train dataset size: {len(train_dataset)}")
    print(f"[+] Test dataset size: {len(test_dataset)}")

    # loading the model

    enc, dec, disc = load_model(
        config,
        device
    )
    print(f"[+] Model loaded successfully")
    

    # We do not compute reconstructions here, we go directly to scoring
    # as some scoring methods (like latent space distance) do not require reconstructions at all.

    print(f"[+] Scoring using method: {config.scoring.method}")

    train_scores, train_labels = score_samples(train_loader, enc, dec, disc, config)
    test_scores, test_labels = score_samples(test_loader, enc, dec, disc, config)

    print(
        f"[+] Scoring completed. Train scores shape: {train_scores.shape}, "
        f"Test scores shape: {test_scores.shape}"
    )


    # threshold calibration


    threshold_model = fit_threshold(train_scores, config)

    prediction = threshold_model.predict(test_scores)
    print(f"[+] Thresholding completed")




    # Evaluation 
    result = {}
    raw_test_labels = test_labels # to save later
    test_labels = prepare_binary_labels(test_labels, getattr(test_dataset, "class_to_idx", None))
    try:
        metrics = ranking_metrics(test_scores.detach().cpu(), test_labels)
    except:
        print("[+] Ranking metrics computation using prediction classification.")
        metrics = ranking_metrics(prediction, test_labels)
    result.update(metrics)
    print(f"[+] Ranking metrics: {result}")

    threshold_metrics_result = threshold_metrics(prediction, test_labels)
    result.update(threshold_metrics_result)
    print(f"[+] Threshold metrics: {threshold_metrics_result}")

    # saving results 

    run_dir = make_run_dir(config, suffix)

    # save config
    save_config_yaml(config, run_dir / "config.yaml")

    # save metrics
    with open(run_dir / "results.json", "w") as f:
        json.dump(result, f, indent=4)



    # save tensors
    torch.save(
        {
            "train_scores": train_scores.detach().cpu(),
            "test_scores": test_scores.detach().cpu(),
            "raw_test_labels": raw_test_labels.detach().cpu(),
            "binary_test_labels": test_labels,
            "prediction": prediction.detach().cpu(),
        },
        run_dir / "outputs.pt",
)

    print(f"[+] Experiment completed. Results saved to: {run_dir}")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()

    main(args.config, args.suffix)