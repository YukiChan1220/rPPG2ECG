"""
Training script for CardioGAN (PyTorch)
- Supports BIDMC dataset format for PPG and ECG signals.
- Implements: LSGAN adversarial losses (MSE), cycle L1 loss, generator/discriminator updates,
  learning-rate linear decay after 'decay_start_epoch'.
"""

import os
import argparse
import time
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model import (build_generators, build_discriminators, magnitude_spectrogram)
from dataset import create_paired_dataloaders


# -------------------------
# Losses
# -------------------------
mse_loss = nn.MSELoss()
l1_loss = nn.L1Loss()


# -------------------------
# Training function
# -------------------------
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    # Build models
    G, F = build_generators(device=device)
    D_time_E, D_spec_E, D_time_P, D_spec_P = build_discriminators(device=device)

    # Optimizers (paper: Adam)
    g_params = list(G.parameters()) + list(F.parameters())
    d_params = list(D_time_E.parameters()) + list(D_spec_E.parameters()) + list(D_time_P.parameters()) + list(D_spec_P.parameters())

    optG = optim.Adam(g_params, lr=args.lr, betas=(0.5, 0.999))
    optD = optim.Adam(d_params, lr=args.lr, betas=(0.5, 0.999))

    # LR schedulers: linear decay after decay_start_epoch to zero at args.epochs
    def lambda_rule(epoch):
        if epoch < args.decay_start_epoch:
            return 1.0
        else:
            return max(0.0, 1.0 - (epoch - args.decay_start_epoch) / (args.epochs - args.decay_start_epoch))

    schedulerG = optim.lr_scheduler.LambdaLR(optG, lr_lambda=lambda_rule)
    schedulerD = optim.lr_scheduler.LambdaLR(optD, lr_lambda=lambda_rule)

    # Create dataloaders from BIDMC dataset
    if args.data_path is None:
        raise ValueError("Please provide --data_path to the directory containing BIDMC CSV files.")

    loader_ppg, loader_ecg = create_paired_dataloaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        segment_length=args.segment_length,
        stride=args.stride,
        target_fs=args.target_fs,
        num_workers=args.num_workers,
        dataset_type=args.dataset_type
    )

    # helper for infinite iterators
    def inf_iter(dl):
        while True:
            for x in dl:
                yield x

    it_ppg = inf_iter(loader_ppg)
    it_ecg = inf_iter(loader_ecg)

    save_dir = Path(args.out_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # labels for LSGAN
    real_label = 1.0
    fake_label = 0.0

    # training loop
    total_steps = 0
    for epoch in range(args.epochs):
        iters_per_epoch = min(len(loader_ppg), len(loader_ecg))
        epoch_start = time.time()
        for i in range(iters_per_epoch):
            total_steps += 1
            # get batches (unpaired)
            ppg = next(it_ppg).to(device)  # (B, 1, L)
            ecg = next(it_ecg).to(device)

            # -------------------------
            # Update discriminators
            # -------------------------
            optD.zero_grad()

            # real ECG domain: D_time_E, D_spec_E
            D_time_E_real = D_time_E(ecg)  # (B,1,L')
            D_spec_E_real = D_spec_E(magnitude_spectrogram(ecg, n_fft=args.n_fft, hop_length=args.hop_length, win_length=args.win_length))
            # generate fake ECG from PPG
            fake_ecg = G(ppg)
            D_time_E_fake = D_time_E(fake_ecg.detach())
            D_spec_E_fake = D_spec_E(magnitude_spectrogram(fake_ecg.detach(), n_fft=args.n_fft, hop_length=args.hop_length, win_length=args.win_length))

            # LSGAN loss (MSE)
            loss_D_time_E = mse_loss(D_time_E_real, torch.full_like(D_time_E_real, real_label, device=device)) + \
                            mse_loss(D_time_E_fake, torch.full_like(D_time_E_fake, fake_label, device=device))
            loss_D_spec_E = mse_loss(D_spec_E_real, torch.full_like(D_spec_E_real, real_label, device=device)) + \
                            mse_loss(D_spec_E_fake, torch.full_like(D_spec_E_fake, fake_label, device=device))

            # real PPG domain discriminators
            D_time_P_real = D_time_P(ppg)
            D_spec_P_real = D_spec_P(magnitude_spectrogram(ppg, n_fft=args.n_fft, hop_length=args.hop_length, win_length=args.win_length))
            fake_ppg = F(ecg)
            D_time_P_fake = D_time_P(fake_ppg.detach())
            D_spec_P_fake = D_spec_P(magnitude_spectrogram(fake_ppg.detach(), n_fft=args.n_fft, hop_length=args.hop_length, win_length=args.win_length))

            loss_D_time_P = mse_loss(D_time_P_real, torch.full_like(D_time_P_real, real_label, device=device)) + \
                            mse_loss(D_time_P_fake, torch.full_like(D_time_P_fake, fake_label, device=device))
            loss_D_spec_P = mse_loss(D_spec_P_real, torch.full_like(D_spec_P_real, real_label, device=device)) + \
                            mse_loss(D_spec_P_fake, torch.full_like(D_spec_P_fake, fake_label, device=device))

            loss_D = args.lambda_time_adv * (loss_D_time_E + loss_D_time_P) + args.lambda_freq_adv * (loss_D_spec_E + loss_D_spec_P)
            loss_D.backward()
            optD.step()

            # -------------------------
            # Update generators
            # -------------------------
            optG.zero_grad()

            # adversarial loss: push generators to fool discriminators (real label)
            D_time_E_pred = D_time_E(fake_ecg)
            D_spec_E_pred = D_spec_E(magnitude_spectrogram(fake_ecg, n_fft=args.n_fft, hop_length=args.hop_length, win_length=args.win_length))
            loss_G_adv_E_time = mse_loss(D_time_E_pred, torch.full_like(D_time_E_pred, real_label, device=device))
            loss_G_adv_E_freq = mse_loss(D_spec_E_pred, torch.full_like(D_spec_E_pred, real_label, device=device))

            D_time_P_pred = D_time_P(fake_ppg)
            D_spec_P_pred = D_spec_P(magnitude_spectrogram(fake_ppg, n_fft=args.n_fft, hop_length=args.hop_length, win_length=args.win_length))
            loss_G_adv_P_time = mse_loss(D_time_P_pred, torch.full_like(D_time_P_pred, real_label, device=device))
            loss_G_adv_P_freq = mse_loss(D_spec_P_pred, torch.full_like(D_spec_P_pred, real_label, device=device))

            # cycle consistency: x -> G(x) -> F(G(x)) ≈ x
            rec_ppg = F(fake_ecg)
            rec_ecg = G(fake_ppg)
            loss_cycle = l1_loss(rec_ppg, ppg) + l1_loss(rec_ecg, ecg)

            # total generator loss: weighted sum
            loss_G = args.lambda_time_adv * (loss_G_adv_E_time + loss_G_adv_P_time) + args.lambda_freq_adv * (loss_G_adv_E_freq + loss_G_adv_P_freq) + args.lambda_cycle * loss_cycle
            loss_G.backward()
            optG.step()

            if total_steps % args.print_every == 0:
                print(f"Epoch[{epoch+1}/{args.epochs}] Step[{i+1}/{iters_per_epoch}] "
                      f"Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f} "
                      f"adv_E_time: {loss_G_adv_E_time.item():.4f} adv_E_freq: {loss_G_adv_E_freq.item():.4f} "
                      f"adv_P_time: {loss_G_adv_P_time.item():.4f} adv_P_freq: {loss_G_adv_P_freq.item():.4f} "
                      f"cycle: {loss_cycle.item():.4f}")

        # schedulers step (linear lr decay)
        schedulerG.step()
        schedulerD.step()

        # save checkpoints
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            torch.save({
                'G': G.state_dict(),
                'F': F.state_dict(),
                'D_time_E': D_time_E.state_dict(),
                'D_spec_E': D_spec_E.state_dict(),
                'D_time_P': D_time_P.state_dict(),
                'D_spec_P': D_spec_P.state_dict(),
                'optG': optG.state_dict(),
                'optD': optD.state_dict(),
                'epoch': epoch
            }, save_dir / f'cardiogan_epoch{epoch+1}.pth')
            print(f"Saved checkpoint at epoch {epoch+1}")

    print("Training finished.")


# -------------------------
# CLI and defaults
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Train CardioGAN on BIDMC dataset')
    
    # Dataset arguments
    parser.add_argument('--data_path', type=str, required=True, 
                       help='Path to directory containing BIDMC CSV files (bidmc_XX_Signals.csv)')
    parser.add_argument('--dataset_type', type=str, default='bidmc', 
                       help='Dataset type (currently supports: bidmc)')
    parser.add_argument('--segment_length', type=int, default=512, 
                       help='Length of each signal segment in samples at target_fs')
    parser.add_argument('--stride', type=int, default=256, 
                       help='Stride for sliding window segmentation (default: 256 for 50%% overlap)')
    parser.add_argument('--target_fs', type=int, default=128, 
                       help='Target sampling frequency in Hz (default: 128)')
    
    # Training arguments
    parser.add_argument('--out_dir', type=str, default='checkpoints', help='output directory')
    parser.add_argument('--epochs', type=int, default=15, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--decay_start_epoch', type=int, default=10, help='epoch to start linear lr decay')
    
    # Spectrogram arguments
    parser.add_argument('--n_fft', type=int, default=256, help='n_fft for STFT')
    parser.add_argument('--hop_length', type=int, default=64, help='hop_length for STFT')
    parser.add_argument('--win_length', type=int, default=256, help='win_length for STFT')
    
    # Loss weights
    parser.add_argument('--lambda_time_adv', type=float, default=3.0, help='adversarial loss weight for D_t')
    parser.add_argument('--lambda_freq_adv', type=float, default=1.0, help='adversarial loss weight for D_f')
    parser.add_argument('--lambda_cycle', type=float, default=30.0, help='cycle consistency weight')
    
    # Logging and saving
    parser.add_argument('--print_every', type=int, default=50, help='print every N steps')
    parser.add_argument('--save_every', type=int, default=10, help='save every N epochs')
    
    # System arguments
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--no_cuda', action='store_true', help='do not use cuda even if available')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
