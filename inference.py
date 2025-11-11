"""
Inference script for CardioGAN - Generate ECG from PPG signals.
"""

import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from model import build_generators


def load_checkpoint(checkpoint_path, device='cpu'):
    """Load trained generator from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    G, F = build_generators(device=device)
    G.load_state_dict(checkpoint['G'])
    F.load_state_dict(checkpoint['F'])
    G.eval()
    F.eval()
    return G, F


def load_ppg_from_csv(csv_path, segment_length=512, normalize=True):
    """Load PPG signal from BIDMC CSV file."""
    df = pd.read_csv(csv_path)
    if 'PLETH' not in df.columns:
        raise ValueError(f"PLETH column not found in {csv_path}")
    
    ppg = df['PLETH'].values
    ppg = ppg[~np.isnan(ppg)]
    
    # Segment into windows
    segments = []
    start_idx = 0
    while start_idx + segment_length <= len(ppg):
        segment = ppg[start_idx:start_idx + segment_length]
        if normalize:
            mean, std = segment.mean(), segment.std()
            if std > 1e-6:
                segment = (segment - mean) / std
        segments.append(segment)
        start_idx += segment_length
    
    return np.array(segments, dtype=np.float32)


def generate_ecg_from_ppg(G, ppg_segments, device='cpu', batch_size=32):
    """Generate ECG from PPG segments using trained generator."""
    G.eval()
    generated_ecgs = []
    
    with torch.no_grad():
        num_segments = len(ppg_segments)
        for i in range(0, num_segments, batch_size):
            batch = ppg_segments[i:i+batch_size]
            batch_tensor = torch.tensor(batch, dtype=torch.float32).unsqueeze(1).to(device)
            fake_ecg = G(batch_tensor)
            generated_ecgs.append(fake_ecg.cpu().numpy())
    
    generated_ecgs = np.concatenate(generated_ecgs, axis=0)
    return generated_ecgs.squeeze(1)  # (N, L)


def save_results(output_path, ppg_segments, generated_ecgs, sample_rate=125):
    """Save PPG and generated ECG to CSV file."""
    # Concatenate all segments (simple concatenation, may have discontinuities)
    ppg_signal = ppg_segments.flatten()
    ecg_signal = generated_ecgs.flatten()
    
    # Create time axis
    time = np.arange(len(ppg_signal)) / sample_rate
    
    # Save to CSV
    df = pd.DataFrame({
        'Time [s]': time,
        'PPG_Input': ppg_signal,
        'ECG_Generated': ecg_signal
    })
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate ECG from PPG using trained CardioGAN')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint (.pth)')
    parser.add_argument('--input_csv', type=str, required=True,
                       help='Path to input BIDMC CSV file containing PPG')
    parser.add_argument('--output_csv', type=str, default='generated_ecg.csv',
                       help='Path to save generated ECG results')
    parser.add_argument('--segment_length', type=int, default=512,
                       help='Segment length (must match training)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference')
    parser.add_argument('--no_cuda', action='store_true',
                       help='Do not use CUDA even if available')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading checkpoint from {args.checkpoint}...")
    G, F = load_checkpoint(args.checkpoint, device=device)
    print("Model loaded successfully!")
    
    # Load PPG signal
    print(f"Loading PPG from {args.input_csv}...")
    ppg_segments = load_ppg_from_csv(args.input_csv, 
                                     segment_length=args.segment_length, 
                                     normalize=True)
    print(f"Loaded {len(ppg_segments)} PPG segments")
    
    # Generate ECG
    print("Generating ECG from PPG...")
    generated_ecgs = generate_ecg_from_ppg(G, ppg_segments, device=device, 
                                           batch_size=args.batch_size)
    print(f"Generated {len(generated_ecgs)} ECG segments")
    
    # Save results
    save_results(args.output_csv, ppg_segments, generated_ecgs)
    print("Done!")


if __name__ == '__main__':
    main()
