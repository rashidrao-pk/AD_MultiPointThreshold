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


1. Dataset Source:
    > https://zenodo.org/records/18742241?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjczM2FjZDQ5LTY4ODAtNGE2YS05MzQzLTNmMTU5NzY2YzE5MCIsImRhdGEiOnt9LCJyYW5kb20iOiJjNjJmZGY4Y2E0ZWI5MzMwMDI5MzE0NzdlZTcwNTZhMyJ9.9w6dITIp2q681wEH31ZCUg5y5hi3rRy60cHxaLixOm1-5xTIkNjldKMaDvmB8hQRYrHoJ7A_nWNm7fWcTe4KPQ


4. 

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


## Step 2: Retrive Datasets
These datasets are used and needs to be downloaded
1. DistriMuSe_UniGra dataset
2. Robotics Hazards
3. MVTec Dataset


| Dataset | Description | LINK |
|---|---| ---- |
| Synthetic Palletizing | Valeria-Lab, University of Granada | [**Zenodo/Kinematics** dataset](https://zenodo.org/records/18742241?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjczM2FjZDQ5LTY4ODAtNGE2YS05MzQzLTNmMTU5NzY2YzE5MCIsImRhdGEiOnt9LCJyYW5kb20iOiJjNjJmZGY4Y2E0ZWI5MzMwMDI5MzE0NzdlZTcwNTZhMyJ9.9w6dITIp2q681wEH31ZCUg5y5hi3rRy60cHxaLixOm1-5xTIkNjldKMaDvmB8hQRYrHoJ7A_nWNm7fWcTe4KPQ)

 > Tool used to generate Synthetic Dataset by Valeria-LAB is available at https://github.com/valerialabugr/SimIndus-Dataset.

## Step 2: Retrive Model Weight

> [!NOTE]
> Model checkpoints are not included in the repository because of file size limitations.

a. Retreive Models from GitLab
> https://gitlab.di.unito.it/rashid/dm_checkpoints_demo32


```bash
cd AD_MultiPointThreshold/models/DistriMuSe_synthetic
git clone https://gitlab.di.unito.it/rashid/dm_checkpoints_demo32 origin-url   # FOR Simulated ROBOT Palletizing - DEMO 3.2
cd ..
```


## Run Scripts

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
python scripts/inference.py --dataset DistriMuSe_UniGra
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
    Distrimuse_UniGra: null      # Will use base_dir/Distrimuse_UniGra
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
  --dataset Distrimuse_UniGra \
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
python scripts/train.py --dataset Distrimuse_UniGra --safety_area ALL
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
  --dataset Distrimuse_UniGra \
  --safety_area RoboArm \
  --checkpoints models/ \
  --threshold_dir results/thresholds/
```

### Inference Parameters
- `--dataset` - Dataset selection: MVtec, Robotics_Hazards, Distrimuse_UniGra
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
ls $DATA_DIR/Distrimuse_UniGra
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
- Dataset switching: MVtec, Robotics_Hazards, Distrimuse_UniGra
- Enhanced configuration management
- Model loader and inspection tools
- Comprehensive documentation

---