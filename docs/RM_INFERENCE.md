## RUN TEST

```bash
python -m experiments.run_experiment --config .\configs\local.example.yaml
```

```bash
python -m experiments.run_vaegan --config configs\mvtec.yaml --force
python -m experiments.run_vaegan --config configs\cobots.yaml --force
python -m experiments.run_vaegan --config configs\hazards.yaml --force
```

```bash
python -m experiments.run_vaegan --config configs/mvtec_mac.yaml --force
python -m experiments.run_vaegan --config configs/cobots_mac.yaml --force
python -m experiments.run_vaegan --config configs/hazards_mac.yaml --force
```

Override config values from CLI without editing YAML:

```bash
# Friendly shortcut for MVTec object/category
python -m experiments.run_vaegan --config configs/mvtec_mac.yaml --category zipper --force

# Generic dotted-key overrides; can be repeated
python -m experiments.run_vaegan --config configs/mvtec_mac.yaml --force \
  --set data.category=hazelnut \
  --set threshold.percentile=99 \
  --set scoring.method=reconstruction_l2
```

## PLOT Results:

```bash
python -m modules.plotting.plot_experiment_results --run_dir results/experiments/E00008
```

```bash
# RUN Web App to see live score and dynamics
python app.py
# visit --> http://127.0.0.1:8000
```
