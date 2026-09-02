## Machine-Specific Configuration

The project supports machine-specific paths without committing personal drive
locations. Keep shared defaults in git, and put personal paths in
`configs/local.yaml` or environment variables.

Start from the example file:

```bash
cp configs/local.example.yaml configs/local.yaml
```

Then edit `configs/local.yaml` for your machine:

```yaml
paths:
  data_root: "/path/to/datasets"
  models_root: "/path/to/models"
  checkpoints_root: "/path/to/checkpoints"
  results_root: "/path/to/results"

datasets:
  MVtec: "/path/to/datasets/MVtec"
  Robotics_Hazards: "/path/to/datasets/Robotics_Hazards"
  Cobots_Synthetic: "/path/to/datasets/Cobots_Synthetic"
```

`configs/local.yaml` is ignored by git, so each collaborator can keep their own
paths. You can also set paths with environment variables:

```bash
export DATA_ROOT=/path/to/datasets
export MODELS_ROOT=/path/to/models
export CHECKPOINTS_ROOT=/path/to/checkpoints
export RESULTS_ROOT=/path/to/results
export VAEGAN_ROOT=/path/to/models/vaegan
```

On Windows PowerShell:

```powershell
$env:DATA_ROOT="E:\Datasets"
$env:MODELS_ROOT="E:\Models"
$env:CHECKPOINTS_ROOT="E:\Checkpoints"
$env:RESULTS_ROOT="E:\Results"
$env:VAEGAN_ROOT="E:\Models\vaegan"
```

At startup, config loading expands environment variables and validates required
dataset/model paths. If a path is missing, the program raises a clear error
showing which config key needs to be fixed.
