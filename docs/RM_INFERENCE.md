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

## PLOT Results:

```bash
python -m modules.plotting.plot_experiment_results --run_dir results/experiments/E00008
```

```bash
# RUN Web App to see live score and dynamics
python app.py
# visit --> http://127.0.0.1:8000
```
