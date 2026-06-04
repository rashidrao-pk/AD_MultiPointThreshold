# Full One-Class Ablation Framework

This extension turns the project into a model/score/threshold ablation framework for anomaly detection.

## Added models

| Type | Model | Learns from anomalies? | Score |
|---|---|---:|---|
| Reconstruction | AE | No | L1/L2 reconstruction map |
| Probabilistic reconstruction | Vanilla VAE | No | reconstruction map + optional latent statistics |
| Adversarial reconstruction | VAE-GAN | No | reconstruction map; GAN loss only during training |
| Feature compactness | Deep SVDD | No | distance from normal center |
| Classical latent one-class | Isolation Forest | No | negative decision score |
| Classical latent one-class | One-Class SVM | No | negative decision score |

PatchCore/PaDiM should be added later as external strong baselines, usually using `anomalib`, because their implementations are larger and use pretrained feature memory banks.

## Expected data layout

Generic one-class layout:

```text
DATA_ROOT/
  train/normal/*.png
  val/normal/*.png                  # optional; train is split if missing
  test/normal/*.png
  test/anomaly/*.png
  ground_truth/anomaly/*.png        # optional pixel masks
```

MVTec-like layout is also supported:

```text
DATA_ROOT/object/train/good
DATA_ROOT/object/test/good
DATA_ROOT/object/test/<defect_type>
DATA_ROOT/object/ground_truth/<defect_type>
```

## Train and evaluate neural baselines

```bash
python scripts/run_ablation.py \
  --data_root data/MVtec/hazelnut \
  --dataset_name mvtec_hazelnut \
  --experiment_name mvtec_hazelnut_v1 \
  --models ae vae vaegan deep_svdd \
  --train \
  --epochs 30 \
  --device auto
```

Outputs:

```text
results/ablation/<experiment_name>/summary_all_models.csv
results/ablation/<experiment_name>/scores_<model>.csv
checkpoints/ablation/<experiment_name>/<model>_64.pt
```

## Run multiple score/threshold ablations

```bash
for score in l1_max l1_mean l1_q95 l2_max; do
  for th in q99 median_mad max; do
    python scripts/run_ablation.py \
      --data_root data/MVtec/hazelnut \
      --dataset_name mvtec_hazelnut \
      --experiment_name mvtec_hazelnut_${score}_${th} \
      --models ae vae vaegan deep_svdd \
      --score_method $score \
      --threshold_method $th \
      --device auto
  done
 done
```

Use `--train` the first time. Without `--train`, the script loads checkpoints.

## Classical one-class baselines on latent embeddings

```bash
python scripts/run_classical_oneclass.py \
  --data_root data/MVtec/hazelnut \
  --dataset_name mvtec_hazelnut \
  --experiment_name mvtec_hazelnut_classical \
  --encoder_model ae \
  --train_encoder \
  --epochs 20 \
  --device auto
```

This adds:

- Isolation Forest
- One-Class SVM

Both use only normal training embeddings.

## Aggregate results across datasets/components

```bash
python scripts/aggregate_ablation_results.py \
  --results_dir results/ablation \
  --metric mp_f1 \
  --out results/ablation/leaderboard.csv
```

## Recommended paper comparison

Use three result tables:

1. **Model ablation:** AE vs VAE vs VAE-GAN vs DeepSVDD vs Isolation Forest vs One-Class SVM.
2. **Score ablation:** L1, L2, mean, max, q95/q99 anomaly-map scores.
3. **Threshold ablation:** max validation threshold, quantile threshold, median+MAD, multi-point threshold.

Decision rule:

```text
Best per dataset/component: highest mp_f1, then image_auroc, then pixel_auroc if masks exist.
Best overall: average rank over all datasets/components.
```
