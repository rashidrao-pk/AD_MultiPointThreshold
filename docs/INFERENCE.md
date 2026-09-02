## RUN TEST

```bash

# python -m experiments.run_experiment --config .\configs\local.example.yaml

```

```bash
python -m experiments.run_vaegan --config configs\mvtec_xn2.yaml --force
python -m experiments.run_vaegan --config configs\cobots_xn2.yaml --force
python -m experiments.run_vaegan --config configs\hazards_xn2.yaml --force
```

```bash
python -m experiments.run_vaegan --config configs/mvtec_mac.yaml --force
python -m experiments.run_vaegan --config configs/cobots_mac.yaml --force
python -m experiments.run_vaegan --config configs/hazards_mac.yaml --force
```

```bash

python -m experiments.run_vaegan --config configs/mvtec_g5.yaml --force
python -m experiments.run_vaegan --config configs/cobots_g5.yaml --force
python -m experiments.run_vaegan --config configs/hazards_g5.yaml --force
```

Override config values from CLI without editing YAML:

```bash
# Friendly shortcut for MVTec object/category
python -m experiments.run_vaegan --config configs/mvtec_mac.yaml --category zipper --force

# Run every known MVTec object with the same config template
python -m experiments.run_vaegan --config configs/mvtec_mac.yaml --dataset MVTec --force

# Run one category while also overriding the dataset name
python -m experiments.run_vaegan --config configs/mvtec_mac.yaml --dataset MVTec --category zipper --force

# Run every known Cobots safety area
python -m experiments.run_vaegan --config configs/cobots_mac.yaml --dataset Cobots_Synthetic --force

# Generic dotted-key overrides; can be repeated
python -m experiments.run_vaegan --config configs/mvtec_mac.yaml --force \
  --set data.category=hazelnut \
  --set threshold.percentile=99 \
  --set scoring.method=reconstruction_l2
```

## DATASET wise experiemnts

````bash
python -m experiments.run_vaegan --config configs/mvtec_mac.yaml --dataset MVTec --force

python -m experiments.run_vaegan --config configs/cobots_mac.yaml --dataset Cobots_Synthetic --force

python -m experiments.run_vaegan --config configs/hazards_mac.yaml --force


## PLOT Results:

```bash
python -m modules.plotting.plot_experiment_results --run_dir results/experiments/E00008
```

Or find the latest completed experiment from the config file's `output.dir` and matching model/data/scoring/threshold settings:

```bash
python -m modules.plotting.plot_experiment_results --config configs/mvtec_mac.yaml
```

```bash
# RUN Web App to see live score and dynamics
python app.py
# visit --> http://127.0.0.1:8000
```
````




## 8. Inference with Different Datasets

### Inference Configuration
```bash
# Robotics Hazards dataset
python utils/scripts/inference.py \
  --dataset Robotics_Hazards \
  --safety_area ALL \
  --checkpoints models/ \
  --threshold_dir results/thresholds/

# MVtec dataset
python utils/scripts/inference.py \
  --dataset MVtec \
  --object hazelnut \
  --checkpoints models/MVtec/ \
  --threshold_dir results/thresholds/

# DistriMuSe synthetic
python utils/scripts/inference.py \
  --dataset Cobots_Synthetic \
  --safety_area RoboArm \
  --checkpoints models/ \
  --threshold_dir results/thresholds/
```

### Inference Parameters
- `--dataset` - Dataset selection: MVtec, Robotics_Hazards, Cobots_Synthetic
- `--safety_area` - Safety area to evaluate: PLeft, PRight, RoboArm, ConvBelt, or ALL
- `--latent_dims` - Must match training latent dimensions
- `--quantile` - Anomaly score quantile threshold
- `--max_frames` - Maximum frames to process (None = all)
- `--verbose_level` - Output verbosity: 0-2
