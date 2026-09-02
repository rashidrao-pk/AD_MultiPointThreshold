# Troubleshooting

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
python utils/scripts/model_loader.py --list

# Verify model checkpoint integrity
python utils/scripts/model_loader.py --model_path models/model_RoboArm_64.pt --test_forward
```

### CUDA/GPU Issues
```bash
# Force CPU training
python utils/scripts/train.py --safety_area RoboArm  # Will auto-detect

# Train on CPU explicitly
python scripts/model_loader.py --safety_area RoboArm --device cpu
```

### Out of Memory
```bash
# Reduce batch size
python utils/scripts/train.py --batch_size 8 --safety_area RoboArm

# Reduce image resolution (modify in config)
# Reduce latent dimensions
python utils/scripts/train.py --latent_dims 32 --safety_area RoboArm
```
