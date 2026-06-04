# Multi-Point threshold for Anomaly Detection


![Python](https://img.shields.io/badge/Python-3.9-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![CUDA](https://img.shields.io/badge/CUDA-11.8-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Research-orange)
<a href="https://github.com/rashidrao-pk/AD_MultiPointThreshold"><img src="https://img.shields.io/github/repo-size/rashidrao-pk/AD_MultiPointThreshold" alt="GitHub repo size"></a>
<a href="https://github.com/rashidrao-pk/AD_MultiPointThreshold/commits/main"><img src="https://img.shields.io/github/commit-activity/t/rashidrao-pk/AD_MultiPointThreshold" alt="GitHub commit activity"></a><a href="https://github.com/rashidrao-pk/AD_MultiPointThreshold/graphs/contributors"><img src="https://img.shields.io/github/contributors/rashidrao-pk/AD_MultiPointThreshold" alt="GitHub contributors"></a>
<a href="https://github.com/rashidrao-pk/AD_MultiPointThreshold/commits/main"><img src="https://img.shields.io/github/last-commit/rashidrao-pk/AD_MultiPointThreshold" alt="GitHub last commit"></a>


Note: 
BASE REPO
> https://github.com/rashidrao-pk/advis_distrimuse_unito


Dataset source files and public model checkpoints are listed in the collapsible download section below.

## 1. Start With Environment

```bash
conda create -n AD python==3.9.18
conda activate AD
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

```bash
git clone https://github.com/rashidrao-pk/AD_MultiPointThreshold
conda activate AD
pip install -r requirements.txt
```



---

## 2. Download Datasets and Model Checkpoints

The project uses external datasets and pretrained checkpoints. They are not stored directly in this repository because of file size limitations.

<details>
<summary><strong>2.1 Install download tools</strong></summary>

Install the Kaggle and Hugging Face CLIs:

```bash
pip install -U kaggle huggingface_hub
# or using conda 
# conda install -c conda-forge huggingface_hub
```
#### verify
```bash
hf --help
```

For Kaggle downloads, make sure your Kaggle API token is configured. Download `kaggle.json` from your Kaggle account settings and place it in:

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

</details>

<details>
<summary><strong>2.2 Download datasets from Kaggle</strong></summary>

Supported public Kaggle datasets:

| Dataset | Kaggle URL | Suggested Local Folder |
|---|---|---|
| Robotics Hazards | https://www.kaggle.com/datasets/rashidrao/robotics-hazards | `data/Robotics_Hazards` |
| Cobots Synthetic / DistriMuSe UniGra | https://www.kaggle.com/datasets/rashidrao/cobots-synthetic/ | `data/Cobots_Synthetic` |

Download and unzip:

```bash
mkdir -p data

kaggle datasets download -d rashidrao/robotics-hazards \
  -p data/Robotics_Hazards \
  --unzip

kaggle datasets download -d rashidrao/cobots-synthetic \
  -p data/Cobots_Synthetic \
  --unzip
```


Recommended folder structure:

```text
data/
├── Robotics_Hazards/
├── Cobots_Synthetic/
└── MVtec/
```

Set the dataset root path:

```bash
export DATA_DIR=$(pwd)/data
```

</details>

<details>
<summary><strong>2.3 Download pretrained checkpoints from Hugging Face</strong></summary>

Pretrained model checkpoints are available here:

| Checkpoint Repository | Dataset / Use Case |
|---|---|
| https://huggingface.co/rashidrao/AD_Cobots_Synthetic | Cobots Synthetic / DistriMuSe UniGra |
| https://huggingface.co/rashidrao/AD_Robotics_Hazards | Robotics Hazards |
| https://huggingface.co/rashidrao/AD_MVTec | MVTec AD |

Download all checkpoint repositories:

```bash
mkdir -p checkpoints

huggingface-cli download rashidrao/AD_Cobots_Synthetic \
  --local-dir checkpoints/AD_Cobots_Synthetic

huggingface-cli download rashidrao/AD_Robotics_Hazards \
  --local-dir checkpoints/AD_Robotics_Hazards

huggingface-cli download rashidrao/AD_MVTec \
  --local-dir checkpoints/AD_MVTec
```

or 


```bash
pip install -U huggingface_hub
```

### Download Cobots Synthetic Checkpoints

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="rashidrao/AD_Cobots_Synthetic",
    local_dir="checkpoints/AD_Cobots_Synthetic"
)
```

### Download Robotics Hazards Checkpoints

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="rashidrao/AD_Robotics_Hazards",
    local_dir="checkpoints/AD_Robotics_Hazards"
)
```

### Download MVTec Checkpoints

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="rashidrao/AD_MVTec",
    local_dir="checkpoints/AD_MVTec"
)
```

Expected checkpoint structure:

```text
checkpoints/
├── AD_Cobots_Synthetic/
│   ├── model_PLeft_64.pt
│   ├── model_PRight_64.pt
│   ├── model_RoboArm_64.pt
│   └── model_ConvBelt_64.pt
├── AD_Robotics_Hazards/
└── AD_MVTec/
```

</details>

<details>
<summary><strong>2.4 Verify downloaded checkpoints</strong></summary>

List available models:

```bash
python scripts/model_loader.py --list
```

Inspect a checkpoint and test a forward pass:

```bash
python scripts/model_loader.py \
  --model_path checkpoints/AD_Cobots_Synthetic/model_RoboArm_64.pt \
  --show_summary \
  --test_forward \
  --device auto
```

</details>

<details>
<summary><strong>2.5 Quick inference with pretrained checkpoints</strong></summary>

Example with the Cobots Synthetic / DistriMuSe UniGra RoboArm model:

```bash
python scripts/inference.py \
  --dataset Cobots_Synthetic \
  --safety_area RoboArm \
  --checkpoints checkpoints/AD_Cobots_Synthetic \
  --static_mask_paths masks/PLeft.png masks/PRight.png masks/RoboArm.png masks/ConvBelt.png \
  --threshold_dir results/thresholds
```

Example with all safety areas:

```bash
python scripts/inference.py \
  --dataset Cobots_Synthetic \
  --safety_area ALL \
  --checkpoints checkpoints/AD_Cobots_Synthetic \
  --static_mask_paths masks/PLeft.png masks/PRight.png masks/RoboArm.png masks/ConvBelt.png \
  --threshold_dir results/thresholds
```

</details>

<details>
<summary><strong>2.6 Notes on thresholds</strong></summary>

Thresholds are dataset-specific and camera/setup-specific. If you change the dataset, camera view, preprocessing, or safety-area masks, recalibrate thresholds before reporting final results.

```bash
python scripts/calibrate_threshold.py \
  --safety_area RoboArm \
  --checkpoints checkpoints/AD_Cobots_Synthetic \
  --threshold_strategy percentile \
  --threshold_percentile 99.0
```

</details>

## 3. Run Scripts

<details>
<summary><strong>Optional: Train models Again?

</strong></summary>


Train a `VAE-GAN` model on one (`PLeft`, `PRight`, `RoboArm`, `ConvBelt`) or all safety areas.

```bash
# Single area (default settings)
python scripts/train.py --safety_area RoboArm
```

```bash
# All areas sequentially
python scripts/train.py --safety_area ALL
```
</details>


## 4. Inference

Run anomaly detection from multiple `input sources` including following input sources;

```bash
# Pre-cropped frames, evaluate against annotations
python scripts/inference.py --dataset MVtec --object hazelnut
python scripts/inference.py --dataset Robotics_Hazards
python scripts/inference.py --dataset Cobots_Synthetic
```

---

## 5. Updated Dataset Configuration

The project now supports seamless dataset switching with configuration support. Update `configs/config.yaml` with your dataset paths:

```yaml
data:
  base_dir: "/path/to/your/datasets"
  paths:
    MVtec: null                  # Will use base_dir/MVtec
    Robotics_Hazards: null       # Will use base_dir/Robotics_Hazards
    Cobots_Synthetic: null      # Will use base_dir/Cobots_Synthetic
```

Or set via environment variable:
```bash
export DATA_DIR=/path/to/your/datasets
```

---

## 6. Training Models

### Quick Start Training

Train on a single safety area:
```bash
# Train on RoboArm area with DistriMuSe dataset (default)
python scripts/train.py --safety_area RoboArm

# Train on specific dataset
python scripts/train.py --dataset Robotics_Hazards --safety_area PLeft
python scripts/train.py --dataset MVtec --safety_area RoboArm
```

### Train All Safety Areas Sequentially
```bash
# Train all 4 safety areas: PLeft, PRight, RoboArm, ConvBelt
python scripts/train.py --safety_area ALL
```

### Advanced Training Options

```bash
# Custom training parameters
python scripts/train.py \
  --dataset Cobots_Synthetic \
  --safety_area RoboArm \
  --epochs 300 \
  --batch_size 32 \
  --latent_dims 128 \
  --learning_rate_enc_dec 0.001 \
  --learning_rate_dis 0.0001 \
  --augmentation_type custom \
  --save_figures \
  --verbose_level 2

# Use different experimental settings
python scripts/train.py --safety_area RoboArm --exp_type E3

# Force rebuild train/val split
python scripts/train.py --safety_area RoboArm --force_rebuild_split
```

### Training Configuration

Key training parameters:
- `--epochs` (default: 200) - Number of training epochs
- `--batch_size` (default: 16) - Batch size for training
- `--latent_dims` (default: 64) - Latent space dimensions
- `--augmentation_type` - Choices: "min", "custom"
- `--exp_type` - Experiment type: "E1", "E2", "E3"
- `--save_figures` - Save reconstruction figures during training
- `--verbose_level` - Verbosity: 0, 1, or 2

### Dataset-Specific Training

```bash
# MVtec dataset
python scripts/train.py --dataset MVtec --safety_area RoboArm --epochs 200

# Robotics Hazards
python scripts/train.py --dataset Robotics_Hazards --safety_area ConvBelt

# DistriMuSe synthetic (default)
python scripts/train.py --dataset Cobots_Synthetic --safety_area ALL
```

---

## 7. Model Inspection and Loading

Use the model loader utility to inspect trained models:

### List Available Models
```bash
# Show all models in checkpoint directory
python scripts/model_loader.py --list

# Show models in custom directory
python scripts/model_loader.py --checkpoint /path/to/models --list
```

### Load and Inspect Specific Model
```bash
# Load model by safety area
python scripts/model_loader.py --safety_area RoboArm --show_summary

# Load specific checkpoint file
python scripts/model_loader.py --model_path models/model_RoboArm_64.pt --show_summary

# Show detailed model architecture
python scripts/model_loader.py --model_path models/model_RoboArm_64.pt \
  --show_summary --verbose --latent_dims 64
```

### Test Model Forward Pass
```bash
# Verify model works with forward pass test
python scripts/model_loader.py --safety_area RoboArm --test_forward

# Test on GPU
python scripts/model_loader.py --model_path models/model_RoboArm_64.pt \
  --test_forward --device cuda
```

### Model Details Output

The model inspection shows:
- **Total Parameters**: All model parameters
- **Trainable Parameters**: Gradient-enabled parameters
- **Architecture**: Complete layer-by-layer structure
- **Input/Output Shapes**: Tensor dimensions at each stage
- **Component Information**:
  - Encoder: Compresses images to latent space
  - Decoder: Reconstructs images from latent space
  - Discriminator: Classifies real vs. fake images

Example output:
```
================================================================================
ENCODER
================================================================================
Total Parameters:      1,234,567
Trainable Parameters:  1,234,567
================================================================================
Architecture:
Encoder(
  (conv1): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
  (conv2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
  ...
  (fc_mu): Linear(in_features=8192, out_features=64)
  (fc_logvar): Linear(in_features=8192, out_features=64)
)
================================================================================
```

---

## 8. Inference with Different Datasets

### Inference Configuration
```bash
# Robotics Hazards dataset
python scripts/inference.py \
  --dataset Robotics_Hazards \
  --safety_area ALL \
  --checkpoints models/ \
  --threshold_dir results/thresholds/

# MVtec dataset  
python scripts/inference.py \
  --dataset MVtec \
  --object hazelnut \
  --checkpoints models/MVtec/ \
  --threshold_dir results/thresholds/

# DistriMuSe synthetic
python scripts/inference.py \
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

---

## 9. Output Structure

Training generates organized outputs:

```
results/
├── models/                          # Trained model checkpoints
│   ├── model_RoboArm_64.pt
│   ├── model_PLeft_64.pt
│   ├── model_PRight_64.pt
│   └── model_ConvBelt_64.pt
├── training/
│   ├── RoboArm_64/                 # Training curves & logs
│   │   ├── history_RoboArm_64.png
│   │   └── log_file_full.txt
│   └── ...
└── monitor/                         # Monitoring visualizations (if --save_figures)
    ├── RoboArm_64/
    │   ├── train_epoch_*.png
    │   └── test_epoch_*.png
    └── ...
```

---

## 10. Troubleshooting

### Dataset Not Found
```bash
# Verify dataset path
export DATA_DIR=/correct/path/to/datasets
ls $DATA_DIR/MVtec
ls $DATA_DIR/Robotics_Hazards
ls $DATA_DIR/Cobots_Synthetic
```

### Model Loading Issues
```bash
# Check available models
python scripts/model_loader.py --list

# Verify model checkpoint integrity
python scripts/model_loader.py --model_path models/model_RoboArm_64.pt --test_forward
```

### CUDA/GPU Issues
```bash
# Force CPU training
python scripts/train.py --safety_area RoboArm  # Will auto-detect

# Train on CPU explicitly
python scripts/model_loader.py --safety_area RoboArm --device cpu
```

### Out of Memory
```bash
# Reduce batch size
python scripts/train.py --batch_size 8 --safety_area RoboArm

# Reduce image resolution (modify in config)
# Reduce latent dimensions
python scripts/train.py --latent_dims 32 --safety_area RoboArm
```

---

## 11. Citation & References

- **Base Repository**: https://github.com/rashidrao-pk/advis_distrimuse_unito
- **DistriMuSe Dataset**: https://zenodo.org/records/18742241
- **Synthetic Dataset Generator**: https://github.com/valerialabugr/SimIndus-Dataset
- **MVTec AD Dataset**: https://www.mvtec.com/company/research/datasets/ad

---

## 12. Version History

- **v1.0** (Current): Multi-dataset support, model inspection utilities
- Dataset switching: MVtec, Robotics_Hazards, Cobots_Synthetic
- Enhanced configuration management
- Model loader and inspection tools
- Comprehensive documentation

---