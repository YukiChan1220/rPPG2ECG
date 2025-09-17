from model import UNet, SimpleConvAE
from dataset import RPPG2ECGDataset
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

def frequency_loss(pred, target, fs=512):
    pred_fft = torch.fft.rfft(pred, dim=-1)
    target_fft = torch.fft.rfft(target, dim=-1)
    return torch.nn.functional.mse_loss(torch.abs(pred_fft), torch.abs(target_fft))

def differentiable_peak_detection(x, fs=512):
    window_size = int(0.6 * fs)
    stride = window_size // 2
    maxpool_result = torch.nn.functional.max_pool1d(
        x.unsqueeze(0) if len(x.shape) == 1 else x,
        kernel_size=window_size,
        stride=stride,
        padding=window_size//2,
        return_indices=True
    )
    peak_values, peak_indices = maxpool_result
    return peak_indices.squeeze()

def rr_intervals(peaks, fs=512):
    if len(peaks) < 2:
        return np.array([])
    rr_intervals = np.diff(peaks) / fs * 1000
    return rr_intervals

def rmssd(x):
    x_np = x.cpu().numpy().squeeze()
    diff = np.diff(x_np)
    return np.sqrt(np.mean(diff**2))


def rr_interval_loss(pred, target, fs=512):
    batch_size = pred.shape[0]
    device = pred.device
    total_loss = torch.tensor(0.0, device=device)
    for i in range(batch_size):
        pred_signal = pred[i].squeeze()
        target_signal = target[i].squeeze()
        window_size = int(0.6 * fs)
        stride = window_size // 4
        pred_maxpool = torch.nn.functional.max_pool1d(
            pred_signal.unsqueeze(0).unsqueeze(0), 
            kernel_size=window_size, 
            stride=stride, 
            padding=window_size//2
        ).squeeze()
        target_maxpool = torch.nn.functional.max_pool1d(
            target_signal.unsqueeze(0).unsqueeze(0), 
            kernel_size=window_size, 
            stride=stride, 
            padding=window_size//2
        ).squeeze()
        if len(pred_maxpool) > 1 and len(target_maxpool) > 1:
            pred_intervals = torch.diff(pred_maxpool)
            target_intervals = torch.diff(target_maxpool)
            
            min_len = min(len(pred_intervals), len(target_intervals))
            if min_len > 0:
                interval_loss = torch.mean((pred_intervals[:min_len] - target_intervals[:min_len]) ** 2)
                total_loss += interval_loss
    return total_loss / batch_size if batch_size > 0 else torch.tensor(0.0, device=device)


def rmssd_loss(pred, target, fs=512):
    device = pred.device
    batch_size = pred.shape[0]
    total_loss = torch.tensor(0.0, device=device)
    for i in range(batch_size):
        pred_signal = pred[i].squeeze()
        target_signal = target[i].squeeze()
        pred_diff = torch.diff(pred_signal)
        target_diff = torch.diff(target_signal)
        pred_rmssd = torch.sqrt(torch.mean(pred_diff ** 2))
        target_rmssd = torch.sqrt(torch.mean(target_diff ** 2))
        total_loss += torch.abs(pred_rmssd - target_rmssd)
    return total_loss / batch_size if batch_size > 0 else torch.tensor(0.0, device=device)

def shape_loss(pred, target):
    target_min, target_max = target.min(), target.max()
    pred_min, pred_max = pred.min(), pred.max()
    min_loss = torch.abs(pred_min - target_min)
    max_loss = torch.abs(pred_max - target_max)
    maxminloss = torch.abs((pred_max / -pred_min) - (target_max / -target_min))
    return min_loss + max_loss + maxminloss

def main():
    device = torch.device("cpu")

    model = UNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    mseloss = torch.nn.MSELoss()
    maeloss = torch.nn.L1Loss()
    def criterion(pred, target):
        mse_loss_weight = 0.4
        mae_loss_weight = 0.3
        freq_loss_weight = 0.1
        rmssd_loss_weight = 0.2
        shape_loss_weight = 0.1
        return (mse_loss_weight * mseloss(pred, target) + 
                mae_loss_weight * maeloss(pred, target) +
                freq_loss_weight * frequency_loss(pred, target) +
                rmssd_loss_weight * rmssd_loss(pred, target) +
                shape_loss_weight * shape_loss(pred, target))
    epochs = 10
    loaders = []

    counter = 0
    datapoints = 0
    for file in os.listdir("./cleaned_data"):
        if file.endswith(".csv") and not file.find("000019") > 0:
            print(f"Loading files {counter} / {len(os.listdir('./cleaned_data'))}")
            counter += 1
            with open(os.path.join("./cleaned_data", file), 'r') as f:
                next(f)  # Skip header
                data = [list(map(float, line.strip().split(','))) for line in f if line.strip()]
                X_rppg = [row[1] for row in data]
                Y_ecg = [row[2] for row in data]
                datapoints += len(X_rppg)
            dataset = RPPG2ECGDataset([X_rppg], [Y_ecg], window_samples=1024)
            loaders.append(torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True))
    print(f"Total {datapoints} data points loaded from {counter} files.")

    for epoch in range(epochs):
        time_loss_weight = 0.7
        freq_loss_weight = 1 - time_loss_weight
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()
        for train_loader in loaders:
            for xr, ye in train_loader:
                xr, ye = xr.to(device), ye.to(device)
                pred = model(xr)
                loss = criterion(pred, ye)
                opt.zero_grad(); loss.backward(); opt.step()

    model.eval()

    with open("./cleaned_data/patient_000019_5.csv", 'r') as f:
        next(f)  # Skip header
        data = [list(map(float, line.strip().split(','))) for line in f if line.strip()]
        X_rppg = [row[1] for row in data]
        Y_ecg = [row[2] for row in data]
        X_tensor = torch.from_numpy(np.array(X_rppg).astype('float32')).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,L)
        Y_tensor = torch.from_numpy(np.array(Y_ecg).astype('float32')).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,L)
        
    with torch.no_grad():
        pred = model(X_tensor)

    plt.figure(figsize=(20, 8))
    plt.plot(Y_tensor.cpu().squeeze().numpy(), label='True ECG', alpha=0.7)
    plt.plot(pred.cpu().squeeze().numpy(), label='Predicted ECG', alpha=0.7)
    plt.legend()
    plt.title("ECG Signal Prediction from RPPG")
    plt.show()

if __name__ == "__main__":
    main()

