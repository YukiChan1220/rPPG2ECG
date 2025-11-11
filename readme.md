# CardioGAN (PyTorch re-implementation)

This repository contains a PyTorch re-implementation of **CardioGAN** (Sarkar & Etemad, AAAI 2021): an attention-based CycleGAN for translating PPG → ECG (and inverse), with **dual discriminators** (time-domain and frequency-domain spectrogram discriminators).

This implementation follows the architectural ideas and training procedure described in the paper:
- Attention U-Net 1D generators (soft attention gates on skip connections)
- Dual discriminators per domain:
  - time-domain 1D convolution discriminator
  - frequency-domain 2D spectrogram discriminator (STFT magnitude as input)
- Adversarial losses (LSGAN / MSE), cycle-consistency L1 loss, and linear LR decay schedule

The original TensorFlow implementation and weights are available from the authors: https://github.com/pritamqu/ppg2ecg-cardiogan and the paper: https://arxiv.org/abs/2010.00104

## Files
- `model.py` : PyTorch models (generators, discriminators, spectrogram utility). Generators use kernel_size=16, stride=2, input shape (B, 1, 512).
- `train.py` : Training loop and CLI. Supports BIDMC dataset format.
- `dataset.py` : Dataset loader for BIDMC format, extensible for future datasets.
- `inference.py` : Script for generating ECG from PPG using trained models.
- `test_setup.py` : Test script to verify dataset loading and model initialization before training.
- `example_train.py` : Example usage and training commands.
- `requirements.txt` : Python dependencies.
- `README.md` : this file.

## Quick start
1. Prepare BIDMC dataset:
   - Place BIDMC CSV files (named `bidmc_XX_Signals.csv`) in a directory.
   - Each CSV should contain columns: `Time [s]`, `RESP`, `PLETH`, `V`, `AVR`, `II`.
   - `PLETH` column contains PPG signal, `II` column contains ECG signal (Lead II).

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Test the setup (recommended before training):
```bash
python test_setup.py --data_path /path/to/bidmc/csv/files
```
This will verify that:
- Dataset files are found and can be loaded
- Signals are correctly extracted and segmented
- Models initialize without errors
- Forward passes work correctly

4. Train the model:
```bash
python train.py --data_path /path/to/bidmc/csv/files --batch_size 32 --epochs 200
```

Available training arguments:
- `--data_path`: Path to directory containing BIDMC CSV files (required)
- `--dataset_type`: Dataset type, currently supports 'bidmc' (default: bidmc)
- `--segment_length`: Length of each signal segment (default: 512)
- `--stride`: Stride for sliding window segmentation (default: 256, 50% overlap)
- `--batch_size`: Batch size (default: 32)
- `--epochs`: Number of training epochs (default: 200)
- `--lr`: Initial learning rate (default: 2e-4)
- `--decay_start_epoch`: Epoch to start linear lr decay (default: 100)
- `--lambda_adv`: Adversarial loss weight (default: 1.0)
- `--lambda_cycle`: Cycle consistency weight (default: 10.0)
- `--out_dir`: Output directory for checkpoints (default: checkpoints)
- `--num_workers`: Number of data loading workers (default: 4)

## Model Architecture
- **Generator**: Attention U-Net with kernel_size=16, stride=2, base_filters=64
  - Input: (B, 1, 512) - batch of 512-point signal segments
  - Output: (B, 1, 512) - generated signal in target domain
  - Encoder depth: 4 layers (64 -> 128 -> 256 -> 512 channels)
  - Attention gates on skip connections for salient feature learning

- **Discriminators**: Dual discriminators per domain (time + frequency)
  - Time Discriminator: 1D CNN with kernel_size=16, stride=2
  - Spectrogram Discriminator: 2D CNN with kernel_size=(7,7), stride=2
  - Filter progression: 64 -> 128 -> 256 -> 512

## Dataset Format
The `dataset.py` module supports BIDMC format and is extensible for future datasets:
- **BIDMC**: CSV files with columns [Time [s], RESP, PLETH, V, AVR, II]
- Automatically segments signals with configurable window length and stride
- Applies z-score normalization per segment
- Returns tensors of shape (1, segment_length)

To add support for other datasets, implement a new dataset class in `dataset.py` following the same interface.

## Inference
After training, use the trained model to generate ECG from new PPG signals:

```bash
python inference.py \
    --checkpoint checkpoints/cardiogan_epoch200.pth \
    --input_csv /path/to/bidmc_01_Signals.csv \
    --output_csv generated_ecg.csv
```

This will:
1. Load the PPG signal from the input CSV file
2. Segment it into 512-point windows
3. Generate ECG for each segment using the trained generator
4. Save both PPG and generated ECG to the output CSV file

## Training Details
- **Loss Functions**:
  - LSGAN adversarial loss (MSE-based) for both time and frequency discriminators
  - Cycle consistency L1 loss to preserve signal characteristics
  - Total generator loss = λ_adv × (L_adv_E + L_adv_P) + λ_cycle × L_cycle

- **Training Strategy**:
  - Unpaired training: PPG and ECG segments don't need to be from the same recording
  - Alternating discriminator and generator updates
  - Linear learning rate decay starting at epoch 100
  - Adam optimizer with β1=0.5, β2=0.999

- **Data Preprocessing**:
  - Sliding window segmentation with configurable stride
  - Z-score normalization per segment
  - No assumptions about sampling rate (works with raw data)
