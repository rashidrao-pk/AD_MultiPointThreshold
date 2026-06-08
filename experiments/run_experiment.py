import argparse
from utils import read_config
import torch

from data import load_data
from models.vaegan import load_model
from modules.scoring import score_samples
from modules.thresholding import fit_threshold
from modules.evaluation import ranking_metrics, threshold_metrics, prepare_binary_labels
def main(config_path):
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
        config.model,
        config.data,
        device
    )
    print(f"[+] Model loaded successfully")
    

    # We do not compute reconstructions here, we go directly to scoring
    # as some scoring methods (like latent space distance) do not require reconstructions at all.


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
    test_labels = prepare_binary_labels(test_labels, getattr(test_dataset, "class_to_idx", None))
    metrics = ranking_metrics(test_scores.detach().cpu(), test_labels)
    result.update(metrics)
    print(f"[+] Ranking metrics: {result}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    main(args.config)