from model import UNet, SimpleConvAE
from dataset import RPPG2ECGDataset
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

def load_mirror_data(data_dir):
    loaders = []
    counter = 0
    datapoints = 0
    for file in os.listdir(data_dir):
        try:
            if file.endswith(".csv") and not file.find("000019") > 0:
                print(f"Loading files {counter} / {len(os.listdir(data_dir))}")
                counter += 1
                with open(os.path.join(data_dir, file), 'r') as f:
                    next(f)  # Skip header
                    data = [list(map(float, line.strip().split(','))) for line in f if line.strip()]
                    X_rppg = [row[1] for row in data]
                    Y_ecg = [row[2] for row in data]
                    datapoints += len(X_rppg)
                dataset = RPPG2ECGDataset([X_rppg], [Y_ecg], window_samples=1024, auto_shift=True)
                loaders.append(torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True))
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue
    print(f"Total {datapoints} data points loaded from {counter} files.")
    return loaders

def eval_mirror_data(device, data_dir):
    with open("{}/patient_000019_5.csv".format(data_dir), 'r') as f:
        next(f)  # Skip header
        data = [list(map(float, line.strip().split(','))) for line in f if line.strip()]
        if len(data) % 4 != 0:
            data = data[:-(len(data) % 4)]
        X_rppg = [row[1] for row in data]
        Y_ecg = [row[2] for row in data]
        X_tensor = torch.from_numpy(np.array(X_rppg).astype('float32')).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,L)
        Y_tensor = torch.from_numpy(np.array(Y_ecg).astype('float32')).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,L)
        return X_tensor, Y_tensor

def load_bidmc_data(data_dir):
    loaders = []
    counter = 0
    datapoints = 0
    for file in os.listdir(data_dir):
        try:
            # if file.endswith("Signals.csv") and not file.find("46") > 0:
            if file.endswith("Signals.csv") and (file.find("45") > 0 or file.find("47") > 0):
                print(f"Loading files {counter} / {len(os.listdir(data_dir))}")
                counter += 1
                df = pd.read_csv(os.path.join(data_dir, file))
                X_rppg = df[' PLETH'].tolist()
                Y_ecg = df[' II'].tolist()
                datapoints += len(X_rppg)
                dataset = RPPG2ECGDataset([X_rppg], [Y_ecg], window_samples=1024, fs = 128, auto_shift=False)
                loaders.append(torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True))
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue
    print(f"Total {datapoints} data points loaded from {counter} files.")
    return loaders

def eval_bidmc_data(device, data_dir):
    df = pd.read_csv(os.path.join(data_dir, "bidmc_46_Signals.csv"))
    X_rppg = df[' PLETH'].tolist()
    Y_ecg = df[' II'].tolist()
    X_rppg = X_rppg[:4096]
    Y_ecg = Y_ecg[:4096]
    X_tensor = torch.from_numpy(np.array(X_rppg).astype('float32')).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,L)
    Y_tensor = torch.from_numpy(np.array(Y_ecg).astype('float32')).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,L)
    return X_tensor, Y_tensor

def main():
    device = torch.device("cpu")
    data_dir = "./test_cleaned"
    data_dir = "./cleaned_data"
    data_dir = "D:/Datasets/bidmc-ppg-and-respiration-dataset-1.0.0/bidmc_csv"

    model = UNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    mseloss = torch.nn.MSELoss()
    maeloss = torch.nn.L1Loss()
    smoothl1loss = torch.nn.SmoothL1Loss()
    def criterion(pred, target):
        return mseloss(pred, target)
    epochs = 30
    
    #loaders = load_mirror_data(data_dir)
    loaders = load_bidmc_data(data_dir)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        running_loss = 0.0
        model.train()
        for train_loader in loaders:
            for xr, ye in train_loader:
                xr, ye = xr.to(device), ye.to(device)
                pred = model(xr)
                loss = criterion(pred, ye)
                opt.zero_grad(); loss.backward(); opt.step()
                running_loss += loss.item()
        print(f"  Training Loss: {running_loss/len(train_loader):.6f}")
    
    # 保存模型
    model.eval()
    print("\nSaving models...")
    
    # 保存 PyTorch 模型
    torch.save(model.state_dict(), "rppg2ecg.pth")
    print("PyTorch model saved to: rppg2ecg.pth")
    
    # 保存 ONNX 模型
    torch.onnx.export(model, torch.randn(1,1,1024).to(device), "rppg2ecg.onnx", 
                     input_names=['input'], output_names=['output'], 
                     dynamic_axes={'input': {2: 'length'}, 'output': {2: 'length'}})
    print("ONNX model saved to: rppg2ecg.onnx")

    #X_tensor, Y_tensor = eval_mirror_data(device, data_dir)
    X_tensor, Y_tensor = eval_bidmc_data(device, data_dir)

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

