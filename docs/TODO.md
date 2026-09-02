# TODO LIST

1. [x] ADD different models in it
   - [x] AE - [RM_TRAINING/AE](/docs/RM_TRAINING.md#added-ae)
   - [x] VAE
   - [x] VAE-GAN
   -[x] ADVIS-VAE-GAN — [RM_TRAINING/ADVIS-VAE-GAN](/docs/RM_TRAINING.md#run-for-advis-vae-gan)
   - [ ] GAN
   - [ ] VitVAEGAN

2. ADD support for different datasets
   - [x] Cobots_UniGra
   - [x] Corridor Hazards
   - [x] MVTec
   - [ ] Cookies
   - [ ] ViSA
   - [ ] with MNITS
   - [ ] dsprites
3. Add different anomaly score functions \_reconstruction_score()/[modules\scoring\reconstruction.py](modules\scoring\reconstruction.py)
   - [x] L1
   - [x] L2
   - [ ] Ravi
   - [ ] SSIM
   - [ ] Perceptual
   - Hybrid

4. Add Threshold Calibration MultiPoint
   - [x] Single Point
   - [ ] Single Point with Anomaly Score Functions instead of MSE...
   - [x] Multi Point
     - [x] q99,...

5. Add Policies for MultiPoint
   - [x] Any
   - [x] Majority
   - [x] all

## Modules

1. [x] Data Handling
   - [x] loading
   - [x] preprocessing
2. [x] Plotting Results
   - [x] dataset stats
   - [x] training dynamics
     - [x] losses over epochs
     - [x] anomaly map evolution
     - [x] score distribution
     - [x] Quality metrics
       - [x] score seperation
       - [x] ranking metrics
       - [x] recall at fixed fpr
       - [x] threshold stability
     - [x] Score Components
     - [x] Loss Balance among components
     - [x] Anomaly Maps for Train/Validation data
3. [x] training
   - Training modules for training models

4. [x] threshold_calibration
   - threshhold calibrattion (only normal, mixed)

5. [x] inference
   - inference of trained model using calibrated threshold

6. [x] Evaluation of results
   - [ ] image AUROC, AP, F1-max;
   - [ ] pixel AUROC, AP, F1-max;
   - [ ] pixel AUPRO.

7. [ ] Comparison with Sota
   - [ ] Dinomaly
   <!-- - ResFlow -->

## PENDING TASKS

1.  [ ] #LINE70/[modules\scoring\reconstruction.py](modules\scoring\reconstruction.py) - reconstruction_quantile_score()
    - TODO: we can also try squared error here, but for now we stick to absolute error as it is more interpretable and less sensitive to outliers.
2.  [ ] add comments in config for training and inference
