"""
Test script to verify dataset loading and model initialization.
Run this before training to ensure everything is set up correctly.
"""

import argparse
import torch
from pathlib import Path

from dataset import create_paired_dataloaders
from model import build_generators, build_discriminators


def test_dataset_loading(data_path, batch_size=4, segment_length=512):
    """Test if dataset loads correctly."""
    print("\n" + "=" * 60)
    print("Testing Dataset Loading")
    print("=" * 60)
    
    try:
        ppg_loader, ecg_loader = create_paired_dataloaders(
            data_path=data_path,
            batch_size=batch_size,
            segment_length=segment_length,
            stride=256,
            normalize=True,
            num_workers=0,  # Use 0 for testing
            dataset_type='bidmc'
        )
        
        print(f"✓ PPG dataset loaded: {len(ppg_loader.dataset)} segments")
        print(f"✓ ECG dataset loaded: {len(ecg_loader.dataset)} segments")
        print(f"✓ Number of batches (PPG): {len(ppg_loader)}")
        print(f"✓ Number of batches (ECG): {len(ecg_loader)}")
        
        # Test getting a batch
        ppg_batch = next(iter(ppg_loader))
        ecg_batch = next(iter(ecg_loader))
        
        print(f"✓ PPG batch shape: {ppg_batch.shape}")
        print(f"✓ ECG batch shape: {ecg_batch.shape}")
        
        assert ppg_batch.shape == (batch_size, 1, segment_length), \
            f"Expected shape ({batch_size}, 1, {segment_length}), got {ppg_batch.shape}"
        assert ecg_batch.shape == (batch_size, 1, segment_length), \
            f"Expected shape ({batch_size}, 1, {segment_length}), got {ecg_batch.shape}"
        
        print("✓ Batch shapes are correct!")
        return True
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return False


def test_model_initialization(segment_length=512):
    """Test if models initialize correctly."""
    print("\n" + "=" * 60)
    print("Testing Model Initialization")
    print("=" * 60)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Build generators
        G, F = build_generators(device=device)
        print(f"✓ Generators initialized")
        
        # Build discriminators
        D_time_E, D_spec_E, D_time_P, D_spec_P = build_discriminators(device=device)
        print(f"✓ Discriminators initialized")
        
        # Test forward pass
        batch_size = 2
        dummy_input = torch.randn(batch_size, 1, segment_length).to(device)
        
        # Generator forward
        fake_ecg = G(dummy_input)
        print(f"✓ Generator G forward pass: {dummy_input.shape} -> {fake_ecg.shape}")
        assert fake_ecg.shape == (batch_size, 1, segment_length), \
            f"Expected output shape ({batch_size}, 1, {segment_length}), got {fake_ecg.shape}"
        
        fake_ppg = F(dummy_input)
        print(f"✓ Generator F forward pass: {dummy_input.shape} -> {fake_ppg.shape}")
        
        # Time discriminator forward
        d_out = D_time_E(dummy_input)
        print(f"✓ Time Discriminator forward pass: {dummy_input.shape} -> {d_out.shape}")
        
        # Test spectrogram discriminator
        from model import magnitude_spectrogram
        spec = magnitude_spectrogram(dummy_input, n_fft=256, hop_length=64, win_length=256)
        spec_out = D_spec_E(spec)
        print(f"✓ Spec Discriminator forward pass: {spec.shape} -> {spec_out.shape}")
        
        print("✓ All models working correctly!")
        return True
        
    except Exception as e:
        print(f"✗ Error initializing models: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Test CardioGAN setup')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to directory containing BIDMC CSV files')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for testing')
    parser.add_argument('--segment_length', type=int, default=512,
                       help='Segment length')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("CardioGAN Setup Test")
    print("=" * 60)
    
    # Check if data path exists
    if not Path(args.data_path).exists():
        print(f"✗ Error: Data path does not exist: {args.data_path}")
        return
    
    print(f"Data path: {args.data_path}")
    
    # Test dataset loading
    dataset_ok = test_dataset_loading(args.data_path, args.batch_size, args.segment_length)
    
    # Test model initialization
    model_ok = test_model_initialization(args.segment_length)
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Dataset loading: {'✓ PASSED' if dataset_ok else '✗ FAILED'}")
    print(f"Model initialization: {'✓ PASSED' if model_ok else '✗ FAILED'}")
    
    if dataset_ok and model_ok:
        print("\n✓ All tests passed! You can now run training.")
        print("  Example: python train.py --data_path", args.data_path)
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
    print("=" * 60)


if __name__ == '__main__':
    main()
