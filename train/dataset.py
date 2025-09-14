import torch
from torch.utils.data import Dataset

class RPPG2ECGDataset(Dataset):
    def __init__(self, X_rppg_list, Y_ecg_list, window_samples):
        self.windows = []
        for r, e in zip(X_rppg_list, Y_ecg_list):
            n = len(r)
            step = window_samples // 2
            for i in range(0, n-window_samples+1, step):
                xr = r[i:i+window_samples]
                ye = e[i:i+window_samples]
                self.windows.append((xr.astype('float32'), ye.astype('float32')))

    def __len__(self): return len(self.windows)
    def __getitem__(self, idx):
        xr, ye = self.windows[idx]
        return torch.from_numpy(xr).unsqueeze(0), torch.from_numpy(ye).unsqueeze(0)  # (1,L)
