# Multi-Point threshold for Anomaly Detection


<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?color=00E5C3&lines=MultiPoint+Threshold+for+Robust+Anomaly+Detection;Threshold+Calibration+%7C+Anomaly+Detection+%7C;Safety-Area+Inference+%7C+Thresholding+%7C+Alert+Publishing;University+of+Torino+%7C+DistriMuSe+Project&center=true&width=900&height=45">
</p>

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
Install Python [using Conda](/docs/Setup.md##install-conda)

```bash
conda create -n AD python==3.9.18 -y
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
export DATA_ROOT=$(pwd)/data
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

hf download rashidrao/AD_Cobots_Synthetic \
  --local-dir checkpoints/AD_Cobots_Synthetic

hf download rashidrao/AD_Robotics_Hazards \
  --local-dir checkpoints/AD_Robotics_Hazards

hf download rashidrao/AD_MVTec \
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
python utils/scripts/model_loader.py --list
```

Inspect a checkpoint and test a forward pass:

```bash
python utils/scripts/model_loader.py \
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
python utils/scripts/inference.py \
  --dataset Cobots_Synthetic \
  --safety_area RoboArm \
  --checkpoints checkpoints/AD_Cobots_Synthetic \
  --static_mask_paths masks/PLeft.png masks/PRight.png masks/RoboArm.png masks/ConvBelt.png \
  --threshold_dir results/thresholds
```

Example with all safety areas:

```bash
python utils/scripts/inference.py \
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
python utils/scripts/calibrate_threshold.py \
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
python utils/scripts/train.py --safety_area RoboArm
```

```bash
# All areas sequentially
python utils/scripts/train.py --safety_area ALL
```
</details>


## 4. Inference

Run anomaly detection from multiple `input sources` including following input sources;

```bash
# Pre-cropped frames, evaluate against annotations
python utils/scripts/inference.py --dataset MVtec --object hazelnut
python utils/scripts/inference.py --dataset Robotics_Hazards
python utils/scripts/inference.py --dataset Cobots_Synthetic
```

---

## 5. Configurations

Check Machine Specific configuration [**setup** here](/docs/CONFIGURATIONS.md#machine-specific-configuration)

---

## 6. Training

Check Model  [**Trainings** here](/docs/TRAINING.md#6-training-models)

---

### 7. Verify and Inspect Model Checkpoints

Check [**model checkpoints** here](/docs/MODEL_CHECKPOINTS.md#7-model-inspection-and-loading)

---

### 7. Inference

Check [**inference here**](/docs/INFERENCE.md#8-inference-with-different-datasets)

---

### 10. Troubleshooting

Got errors? check [**Troubleshooting Guides here**](/docs/INFERENCE.md#8-inference-with-different-datasets)

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


# 👥 Contributing

We welcome contributions! Check out our [Contributing Guide](CONTRIBUTING.md) to get started.

<p align="center">
  <a href="https://github.com/rashidrao-pk/AD_MultiPointThreshold/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=rashidrao-pk/AD_MultiPointThreshold" alt="Contributors to AD/MultiPointThreshold" />
  </a>
</p>

<p align="center">
  <b>Thank you to all our contributors!</b>
</p>
