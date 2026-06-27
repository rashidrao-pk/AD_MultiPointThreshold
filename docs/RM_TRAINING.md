## RUN TEST

```bash
python -m utils.scripts.train --config configs/mvtec_mac.yaml

# want to override configs? change it

python -m utils.scripts.train --config configs/mvtec_mac.yaml --epochs 1000

```

## Force Training with Training Curves

```bash

python -m modules.training.train --config configs/mvtec_mac.yaml --force --plot_curves --plot_every 5
```

## Force Training with latent space

```bash

python -m modules.training.train --config configs/mvtec_mac.yaml --force --plot_latent_space
```

## Force Training with Training Curves and latent space

```bash
python -m modules.training.train --config configs/mvtec_mac.yaml --force --plot_curves --plot_every 10 --plot_latent_space

python -m modules.training.train --config configs/mvtec_mac.yaml --force --plot_curves --plot_latent_space --plot_every 5 --latent_space_classes normal

python -m modules.training.train --config configs/mvtec_mac.yaml --force --plot_curves --plot_latent_space  --plot_every 5 --latent_space_classes both

python -m modules.training.train --config configs/mvtec_mac.yaml --force --plot_curves --plot_latent_space --latent_space_classes both --plot_every 5

```

## Choose Latent Project types:

```bash
python -m modules.training.train \
  --config configs/mvtec_mac.yaml \
  --force \
  --plot_curves \
  --plot_latent_space \
  --latent_space_classes both \
  --latent_projection tsne \
  --plot_every 5
```

Use PCA:

```bash
python -m modules.training.train \
  --config configs/mvtec_mac.yaml \
  --force \
  --plot_curves \
  --plot_score_distribution \
  --plot_latent_space \
  --latent_space_classes both \
  --latent_projection pca
```

### RUN FOR ADVIS-VAE-GAN

```bash
python -m modules.training.train --config configs/mvtec_mac.yaml --model vaegan \
  --force \
  --plot_curves \
  --plot_score_distribution \
  --plot_latent_space \
  --latent_space_classes both \
  --latent_projection pca
```

For all Models

```bash
python -m modules.training.train --config configs/mvtec_mac.yaml --model all
```

```bash
python -m modules.training.train --config configs/mvtec_mac.yaml --model all --epochs 1 --dry_run
```
