


## RUN TEST
```bash
python -m experiments.run_experiment --config .\configs\local.example.yaml
```

```bash
python -m experiments.run_vaegan --config configs\mvtech.yaml --force
python -m experiments.run_vaegan --config configs\cobots.yaml --force
python -m experiments.run_vaegan --config configs\hazards.yaml --force

```

## PLOT Results:

```bash
python scripts/plot_experiment_results.py --run_dir results/experiments/E00001
```