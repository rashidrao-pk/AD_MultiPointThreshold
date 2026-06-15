# Download Datasets and Model Checkpoints

This file provides a compact reference for downloading datasets and pretrained checkpoints used by `AD_MultiPointThreshold`.

<details>
<summary><strong>Install Kaggle and Hugging Face download tools</strong></summary>

```bash
pip install -U kaggle huggingface_hub
```

Configure Kaggle API credentials:

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

</details>

<details>
<summary><strong>Download datasets</strong></summary>

```bash
mkdir -p data

kaggle datasets download -d rashidrao/robotics-hazards \
  -p data/Robotics_Hazards \
  --unzip

kaggle datasets download -d rashidrao/cobots-synthetic \
  -p data/Distrimuse_UniGra \
  --unzip

export DATA_DIR=$(pwd)/data
```

Sources:

- Robotics Hazards: https://www.kaggle.com/datasets/rashidrao/robotics-hazards
- Cobots Synthetic / DistriMuSe UniGra: https://www.kaggle.com/datasets/rashidrao/cobots-synthetic/

</details>

<details>
<summary><strong>Download pretrained checkpoints</strong></summary>

```bash
mkdir -p checkpoints

huggingface-cli download rashidrao/AD_Cobots_Synthetic \
  --local-dir checkpoints/AD_Cobots_Synthetic

huggingface-cli download rashidrao/AD_Robotics_Hazards \
  --local-dir checkpoints/AD_Robotics_Hazards

huggingface-cli download rashidrao/AD_MVTec \
  --local-dir checkpoints/AD_MVTec
```

Sources:

- https://huggingface.co/rashidrao/AD_Cobots_Synthetic
- https://huggingface.co/rashidrao/AD_Robotics_Hazards
- https://huggingface.co/rashidrao/AD_MVTec

</details>

<details>
<summary><strong>Verify checkpoints</strong></summary>

```bash
python scripts/model_loader.py --list --checkpoint checkpoints/AD_Cobots_Synthetic

python scripts/model_loader.py \
  --model_path checkpoints/AD_Cobots_Synthetic/model_RoboArm_64.pt \
  --show_summary \
  --test_forward \
  --device auto
```

</details>
