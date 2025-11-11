"""
Dataset loader for CardioGAN training.
Supports BIDMC dataset format with extensible design for future datasets.
"""

import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class BIDMCDataset(Dataset):
    """
    Dataset loader for BIDMC format.
    
    Expected file structure:
    - Files named bidmc_XX_Signals.csv in the data_path directory
    - Each CSV contains columns: Time [s], RESP, PLETH, V, AVR, II
    - PLETH column: PPG signal
    - II column: ECG signal (Lead II)
    
    Args:
        data_path: Path to directory containing BIDMC CSV files
        signal_type: 'ppg' or 'ecg' to specify which signal to load
        segment_length: Length of each segment (default: 512)
        stride: Stride for sliding window segmentation (default: 256, 50% overlap)
        normalize: Whether to apply z-score normalization per segment
    """
    
    def __init__(self, data_path, signal_type='ppg', segment_length=512, 
                 stride=256, normalize=True):
        super().__init__()
        self.data_path = Path(data_path)
        self.signal_type = signal_type.lower()
        self.segment_length = segment_length
        self.stride = stride
        self.normalize = normalize
        
        if self.signal_type not in ['ppg', 'ecg']:
            raise ValueError("signal_type must be 'ppg' or 'ecg'")
        
        # Load all BIDMC files
        self.segments = self._load_and_segment_data()
        
        print(f"Loaded {len(self.segments)} {signal_type.upper()} segments from {data_path}")
    
    def _load_and_segment_data(self):
        """Load all BIDMC CSV files and segment them into fixed-length windows."""
        all_segments = []
        
        # Find all BIDMC signal files
        csv_files = sorted(glob.glob(str(self.data_path / "bidmc_*_Signals.csv")))
        
        if len(csv_files) == 0:
            raise FileNotFoundError(f"No BIDMC CSV files found in {self.data_path}")
        
        for csv_file in csv_files:
            try:
                # Read CSV file
                df = pd.read_csv(csv_file)
                
                # Extract the appropriate signal column
                if self.signal_type == 'ppg':
                    if ' PLETH' not in df.columns:
                        print(f"Warning: PLETH column not found in {csv_file}, skipping...")
                        continue
                    signal = df[' PLETH'].values
                elif self.signal_type == 'ecg':
                    if ' II' not in df.columns:
                        print(f"Warning: II column not found in {csv_file}, skipping...")
                        continue
                    signal = df[' II'].values
                
                # Remove NaN values
                signal = signal[~np.isnan(signal)]
                
                if len(signal) < self.segment_length:
                    print(f"Warning: Signal in {csv_file} too short ({len(signal)} < {self.segment_length}), skipping...")
                    continue
                
                # Segment the signal using sliding window
                segments = self._segment_signal(signal)
                all_segments.extend(segments)
                
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
                continue
        
        if len(all_segments) == 0:
            raise ValueError(f"No valid segments extracted from {self.data_path}")
        
        return np.array(all_segments, dtype=np.float32)
    
    def _segment_signal(self, signal):
        """Segment a continuous signal into fixed-length windows with stride."""
        segments = []
        start_idx = 0
        
        while start_idx + self.segment_length <= len(signal):
            segment = signal[start_idx:start_idx + self.segment_length]
            segments.append(segment)
            start_idx += self.stride
        
        return segments
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        segment = self.segments[idx].copy()
        
        # Apply normalization if requested
        if self.normalize:
            mean = segment.mean()
            std = segment.std()
            if std > 1e-6:  # Avoid division by zero
                segment = (segment - mean) / std
        
        # Convert to tensor with shape (1, L)
        segment = torch.tensor(segment, dtype=torch.float32).unsqueeze(0)
        
        return segment


class MultiDatasetLoader:
    """
    Extensible loader that can handle multiple dataset formats.
    Future datasets can be added by implementing their specific loader methods.
    """
    
    @staticmethod
    def load_dataset(dataset_type, **kwargs):
        """
        Factory method to load different dataset types.
        
        Args:
            dataset_type: Type of dataset ('bidmc', future: 'mimic', 'capno', etc.)
            **kwargs: Dataset-specific arguments
        
        Returns:
            Dataset object
        """
        dataset_type = dataset_type.lower()
        
        if dataset_type == 'bidmc':
            return BIDMCDataset(**kwargs)
        else:
            raise NotImplementedError(f"Dataset type '{dataset_type}' not implemented yet. "
                                    f"Currently supported: 'bidmc'")


def create_paired_dataloaders(data_path, batch_size=32, segment_length=512, 
                              stride=256, normalize=True, num_workers=4,
                              dataset_type='bidmc'):
    """
    Convenience function to create both PPG and ECG dataloaders.
    
    Args:
        data_path: Path to dataset directory
        batch_size: Batch size for training
        segment_length: Length of each segment
        stride: Stride for segmentation
        normalize: Whether to normalize segments
        num_workers: Number of data loading workers
        dataset_type: Type of dataset to load
    
    Returns:
        ppg_loader, ecg_loader: DataLoader objects for PPG and ECG
    """
    from torch.utils.data import DataLoader
    
    # Create datasets
    ppg_dataset = MultiDatasetLoader.load_dataset(
        dataset_type,
        data_path=data_path,
        signal_type='ppg',
        segment_length=segment_length,
        stride=stride,
        normalize=normalize
    )
    
    ecg_dataset = MultiDatasetLoader.load_dataset(
        dataset_type,
        data_path=data_path,
        signal_type='ecg',
        segment_length=segment_length,
        stride=stride,
        normalize=normalize
    )
    
    # Create dataloaders
    ppg_loader = DataLoader(
        ppg_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    ecg_loader = DataLoader(
        ecg_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return ppg_loader, ecg_loader
