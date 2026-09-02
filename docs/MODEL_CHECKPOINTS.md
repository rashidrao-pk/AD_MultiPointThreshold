## 7. Model Inspection and Loading

Use the model loader utility to inspect trained models:

### List Available Models
```bash
# Show all models in checkpoint directory
python utils/scripts/model_loader.py --list

# Show models in custom directory
python utils/scripts/model_loader.py --checkpoint /path/to/models --list
```

### Load and Inspect Specific Model
```bash
# Load model by safety area
python utils/scripts/model_loader.py --safety_area RoboArm --show_summary

# Load specific checkpoint file
python utils/scripts/model_loader.py --model_path models/model_RoboArm_64.pt --show_summary

# Show detailed model architecture
python utils/scripts/model_loader.py --model_path models/model_RoboArm_64.pt \
  --show_summary --verbose --latent_dims 64
```

### Test Model Forward Pass
```bash
# Verify model works with forward pass test
python utils/scripts/model_loader.py --safety_area RoboArm --test_forward

# Test on GPU
python utils/scripts/model_loader.py --model_path models/model_RoboArm_64.pt \
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
