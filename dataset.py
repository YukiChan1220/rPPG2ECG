"""
Dataset loader for CardioGAN training.
Supports BIDMC dataset format with extensible design for future datasets.

Preprocessing pipeline:
- Resample to 128Hz
- Bandpass FIR filter (3Hz, 45Hz) for ECG
- Bandpass Butterworth filter (1Hz, 8Hz) for PPG
- Z-score normalization for full signal
- Min-max [-1, 1] normalization per segment
"""

import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy import signal
from scipy.interpolate import interp1d


class BIDMCDataset(Dataset):
    """
    Dataset loader for BIDMC format with preprocessing.
    
    Expected file structure:
    - Files named bidmc_XX_Signals.csv in the data_path directory
    - Each CSV contains columns: Time [s], RESP, PLETH, V, AVR, II
    - PLETH column: PPG signal
    - II column: ECG signal (Lead II)
    
    Preprocessing pipeline:
    1. Resample to 128Hz
    2. Apply bandpass filter (FIR for ECG: 3-45Hz, Butterworth for PPG: 1-8Hz)
    3. Z-score normalization on full signal
    4. Segmentation with sliding window
    5. Min-max [-1, 1] normalization per segment
    
    Args:
        data_path: Path to directory containing BIDMC CSV files
        signal_type: 'ppg' or 'ecg' to specify which signal to load
        segment_length: Length of each segment in samples at 128Hz (default: 512)
        stride: Stride for sliding window segmentation (default: 256, 50% overlap)
        target_fs: Target sampling frequency in Hz (default: 128)
    """
    
    def __init__(self, data_path, signal_type='ppg', segment_length=512, 
                 stride=256, target_fs=128):
        super().__init__()
        self.data_path = Path(data_path)
        self.signal_type = signal_type.lower()
        self.segment_length = segment_length
        self.stride = stride
        self.target_fs = target_fs
        
        if self.signal_type not in ['ppg', 'ecg']:
            raise ValueError("signal_type must be 'ppg' or 'ecg'")
        
        # Load all BIDMC files with preprocessing
        self.segments = self._load_and_segment_data()
        
        print(f"Loaded {len(self.segments)} {signal_type.upper()} segments from {data_path}")
    
    def _resample_signal(self, sig, time_values, target_fs):
        """Resample signal to target frequency using interpolation."""
        # Calculate original sampling frequency
        time_diff = np.diff(time_values)
        orig_fs = 1.0 / np.median(time_diff)
        
        if abs(orig_fs - target_fs) < 0.1:  # Already at target frequency
            return sig
        
        # Create interpolation function
        f = interp1d(time_values, sig, kind='cubic', bounds_error=False, fill_value='extrapolate')
        
        # Create new time axis at target frequency
        duration = time_values[-1] - time_values[0]
        n_samples = int(duration * target_fs)
        new_time = np.linspace(time_values[0], time_values[-1], n_samples)
        
        # Resample
        resampled = f(new_time)
        return resampled
    
    def _apply_bandpass_filter(self, sig, fs):
        """Apply appropriate bandpass filter based on signal type."""
        if self.signal_type == 'ecg':
            # FIR bandpass filter for ECG: 3-45 Hz
            numtaps = 101  # Filter order
            lowcut = 3.0
            highcut = 45.0
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            
            # Design FIR filter
            fir_coeff = signal.firwin(numtaps, [low, high], pass_zero=False)
            filtered = signal.filtfilt(fir_coeff, 1.0, sig)
            
        elif self.signal_type == 'ppg':
            # Butterworth bandpass filter for PPG: 1-8 Hz
            lowcut = 1.0
            highcut = 8.0
            order = 4
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            
            # Design Butterworth filter
            b, a = signal.butter(order, [low, high], btype='band')
            filtered = signal.filtfilt(b, a, sig)
        
        return filtered
    
    def _apply_zscore_normalization(self, sig):
        """Apply z-score normalization to the full signal."""
        mean = np.mean(sig)
        std = np.std(sig)
        if std > 1e-6:
            normalized = (sig - mean) / std
        else:
            normalized = sig - mean
        return normalized
    
    def _load_and_segment_data(self):
        """Load all BIDMC CSV files, preprocess, and segment them into fixed-length windows."""
        all_segments = []
        
        # Find all BIDMC signal files
        csv_files = sorted(glob.glob(str(self.data_path / "bidmc_*_Signals.csv")))
        
        if len(csv_files) == 0:
            raise FileNotFoundError(f"No BIDMC CSV files found in {self.data_path}")
        
        for csv_file in csv_files:
            try:
                # Read CSV file
                df = pd.read_csv(csv_file)
                
                # Extract time and signal columns
                if 'Time [s]' not in df.columns:
                    print(f"Warning: 'Time [s]' column not found in {csv_file}, skipping...")
                    continue
                
                time_values = df['Time [s]'].values
                
                # Extract the appropriate signal column
                if self.signal_type == 'ppg':
                    if ' PLETH' not in df.columns:
                        print(f"Warning: PLETH column not found in {csv_file}, skipping...")
                        continue
                    raw_signal = df[' PLETH'].values
                elif self.signal_type == 'ecg':
                    if ' II' not in df.columns:
                        print(f"Warning: II column not found in {csv_file}, skipping...")
                        continue
                    raw_signal = df[' II'].values
                
                # Remove NaN values
                valid_idx = ~np.isnan(raw_signal) & ~np.isnan(time_values)
                raw_signal = raw_signal[valid_idx]
                time_values = time_values[valid_idx]
                
                if len(raw_signal) < 100:  # Need minimum samples for processing
                    print(f"Warning: Signal in {csv_file} too short after NaN removal, skipping...")
                    continue
                
                # Preprocessing pipeline
                # 1. Resample to target frequency (128Hz)
                resampled = self._resample_signal(raw_signal, time_values, self.target_fs)
                
                # 2. Apply bandpass filter
                filtered = self._apply_bandpass_filter(resampled, self.target_fs)
                
                # 3. Apply z-score normalization on full signal
                normalized = self._apply_zscore_normalization(filtered)
                
                # Check if signal is long enough for segmentation
                if len(normalized) < self.segment_length:
                    print(f"Warning: Signal in {csv_file} too short after preprocessing "
                          f"({len(normalized)} < {self.segment_length}), skipping...")
                    continue
                
                # 4. Segment the signal using sliding window
                segments = self._segment_signal(normalized)
                all_segments.extend(segments)
                
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
                import traceback
                traceback.print_exc()
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
        
        # Apply min-max [-1, 1] normalization per segment
        seg_min = segment.min()
        seg_max = segment.max()
        
        if seg_max - seg_min > 1e-6:  # Avoid division by zero
            # Normalize to [-1, 1]
            segment = 2 * (segment - seg_min) / (seg_max - seg_min) - 1
        else:
            # If segment is constant, set to 0
            segment = np.zeros_like(segment)
        
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
                              stride=256, target_fs=128, num_workers=4,
                              dataset_type='bidmc'):
    """
    Convenience function to create both PPG and ECG dataloaders.
    
    Preprocessing is automatically applied:
    - Resample to target_fs (default 128Hz)
    - Bandpass filtering (FIR 3-45Hz for ECG, Butterworth 1-8Hz for PPG)
    - Z-score normalization on full signal
    - Min-max [-1, 1] normalization per segment
    
    Args:
        data_path: Path to dataset directory
        batch_size: Batch size for training
        segment_length: Length of each segment in samples at target_fs
        stride: Stride for segmentation
        target_fs: Target sampling frequency in Hz (default: 128)
        num_workers: Number of data loading workers
        dataset_type: Type of dataset to load
    
    Returns:
        ppg_loader, ecg_loader: DataLoader objects for PPG and ECG
    """
    from torch.utils.data import DataLoader
    
    # Create datasets with preprocessing
    ppg_dataset = MultiDatasetLoader.load_dataset(
        dataset_type,
        data_path=data_path,
        signal_type='ppg',
        segment_length=segment_length,
        stride=stride,
        target_fs=target_fs
    )
    
    ecg_dataset = MultiDatasetLoader.load_dataset(
        dataset_type,
        data_path=data_path,
        signal_type='ecg',
        segment_length=segment_length,
        stride=stride,
        target_fs=target_fs
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
