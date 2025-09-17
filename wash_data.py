import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import os
import pandas as pd

def load_signal_from_merged_log(file_path, data_col):
    time = []
    raw_signal = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                time.append(float(parts[0]))
                if parts[data_col] != '':
                    raw_signal.append(float(parts[data_col]))  # 2-ECG, 1-RPPG
    print(f"Loaded {len(raw_signal)} data points from {file_path}")
    return np.array(time), np.array(raw_signal)


def clean_signal(sig: np.ndarray, fs, config: dict):
    mask = np.ones(len(sig), dtype=bool)
    mask[:] = True
    if len(sig) == 0:
        mask[:] = False
        return mask

    if "std" in config:
        window_len = int(config.get("std").get("window_size") * fs)
        threshold = config.get("std").get("threshold")
        global_std = np.std(sig)
        for start in range(0, len(sig)-window_len, window_len):
            seg = sig[start:start+window_len]
            seg_std = np.std(seg)
            if seg_std > global_std * threshold:
                mask[start:start+window_len] = False

    if "diff" in config:
        window_len = int(config.get("diff").get("window_size") * fs)
        threshold = config.get("diff").get("threshold")
        global_diff = np.max(sig) - np.min(sig)
        for start in range(0, len(sig)-window_len, window_len):
            seg = sig[start:start+window_len]
            seg_diff = np.max(seg) - np.min(seg)
            if seg_diff > global_diff * threshold:
                mask[start:start+window_len] = False

    if "welch" in config:
        window_len = int(config.get("welch").get("window_size") * fs)
        freq_tolerance = config.get("welch").get("bpm_tolerance") / 60
        gf, gPxx = signal.welch(sig, fs=fs, nperseg=window_len)
        peak_freq = gf[np.argmax(gPxx)]
        for start in range(0, len(sig)-window_len, window_len):
            seg = sig[start:start+window_len]
            f, Pxx = signal.welch(seg, fs=fs, nperseg=window_len)
            seg_peak_freq = f[np.argmax(Pxx)]
            if abs(seg_peak_freq - peak_freq) > freq_tolerance:
                mask[start:start+window_len] = False


    return mask


def plot_signals(ecg_ax, rppg_ax, time, ecg_signal, ecg_mask, rppg_signal, rppg_mask, event_handler):
    ecg_ax.plot(time, ecg_signal, label='ECG Signal', alpha=0.7)
    ecg_ax.fill_between(time, ecg_signal, where=~ecg_mask, color='red', alpha=0.7, label='Marked Artifacts')
    ecg_ax.set_xlabel('Time')
    ecg_ax.set_ylabel('Amplitude')
    ecg_ax.set_title('ECG Signal')
    ecg_ax.legend()
    ecg_ax.grid()

    rppg_ax.plot(time, rppg_signal, label='RPPG Signal', alpha=0.7)
    rppg_ax.fill_between(time, rppg_signal, where=~rppg_mask, color='red', alpha=0.7, label='Marked Artifacts')
    rppg_ax.set_xlabel('Time')
    rppg_ax.set_ylabel('Amplitude')
    rppg_ax.set_title('RPPG Signal')
    rppg_ax.legend()
    rppg_ax.grid()

    plt.draw()
    while not event_handler.status:
        plt.pause(0.1)
    ecg_ax.cla()
    rppg_ax.cla()
    event_handler.status = False
    return event_handler.accept, event_handler.reverse



class PltEventHandler:
    def __init__(self):
        self.status = False
        self.accept = False
        self.reverse = False

    def accept_handler(self):
        def handler(event):
            self.status = True
            self.accept = True
            self.reverse = False
        return handler

    def reject_handler(self):
        def handler(event):
            self.status = True
            self.accept = False
            self.reverse = False
        return handler
    
    def reverse_handler(self):
        def handler(event):
            self.status = True
            self.accept = True
            self.reverse = True
        return handler

def log_cleaned_data(file_path, time, ecg_signal, rppg_signal, rppg_mask, ecg_mask):
    clean_windows = []
    window_begin = 0
    window_end = 0
    file_idx = 0

    while window_begin < len(time):
        if rppg_mask[window_begin] and ecg_mask[window_begin]:
            while window_end < len(time) and rppg_mask[window_end] and ecg_mask[window_end]:
                window_end += 1
            clean_windows.append((window_begin, window_end))
        if window_end <= window_begin:
            window_end = window_begin + 1
        window_begin = window_end + 1

    for idx in range(len(clean_windows)):
        start, end = clean_windows[idx]
        if start < end:
            with open(file_path.replace('.csv', f'_{file_idx+1}.csv'), 'w') as f:
                f.write("Time,rPPG,ECG\n")
                for i in range(start, end):
                    f.write(f"{time[i]},{rppg_signal[i]},{ecg_signal[i]}\n")
                file_idx += 1

    print(f"Cleaned data logged to {file_path}")

def show_cleaned_data(file_path):
    fig, (ecg_ax, rppg_ax) = plt.subplots(2, 1, figsize=(20, 12))
    plt.subplots_adjust(bottom=0.15)
    event_handler = PltEventHandler()
    accept = plt.axes([0.7, 0.05, 0.1, 0.05])
    reject = plt.axes([0.81, 0.05, 0.1, 0.05])
    reverse = plt.axes([0.59, 0.05, 0.1, 0.05])
    baccept = plt.Button(accept, 'Accept')
    breject = plt.Button(reject, 'Reject')
    breverse = plt.Button(reverse, 'Reverse')
    baccept.on_clicked(event_handler.accept_handler())
    breject.on_clicked(event_handler.reject_handler())
    breverse.on_clicked(event_handler.reverse_handler())

    for path in os.listdir(file_path):
        if os.path.isfile(os.path.join(file_path, path)) and path.endswith('.csv'):
            try:
                time, rppg_signal = load_signal_from_merged_log(os.path.join(file_path, path), 1)
                _, ecg_signal = load_signal_from_merged_log(os.path.join(file_path, path), 2)
                mask = np.ones(len(rppg_signal), dtype=bool)
                mask[:] = True
                plt.ion()
                accepted, reversed = plot_signals(ecg_ax, rppg_ax, time, ecg_signal, mask, rppg_signal, mask, event_handler)
                if not accepted:
                    os.remove(os.path.join(file_path, path))
                    print(f"Data in {path} rejected by user.")
                else:
                    # 标准差归一化
                    if reversed:
                        ecg_signal = -ecg_signal
                    rppg_signal = (rppg_signal - np.mean(rppg_signal)) / np.std(rppg_signal)
                    ecg_signal = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)
                    df = pd.DataFrame({
                        'Time': time,
                        'rPPG': rppg_signal,
                        'ECG': ecg_signal
                    })
                    df.to_csv(os.path.join(file_path, path), index=False)
                    print(f"Data in {path} accepted and normalized by user.")

            except Exception as e:
                print(f"Error displaying {path}: {e}")
                continue
    
    plt.close()


def clean_data(patient_data_path, fs=512):
    ecg_clean_config = {
        "std": {
            "window_size": 1,
            "threshold": 1.5
        },
    }

    rppg_clean_config = {
        "std": {
            "window_size": 1,
            "threshold": 1.5
        },
        "welch": {
            "window_size": 5,
            "bpm_tolerance": 15
        }
    }

    plt.ion()
    fig, (ecg_ax, rppg_ax) = plt.subplots(2, 1, figsize=(20, 12))
    plt.subplots_adjust(bottom=0.15)
    event_handler = PltEventHandler()
    accept = plt.axes([0.7, 0.05, 0.1, 0.05])
    reject = plt.axes([0.81, 0.05, 0.1, 0.05])
    baccept = plt.Button(accept, 'Accept')
    breject = plt.Button(reject, 'Reject')
    baccept.on_clicked(event_handler.accept_handler())
    breject.on_clicked(event_handler.reject_handler())


    # 逐一人工检查数据
    for dir in os.listdir(patient_data_path):
        if os.path.isdir(os.path.join(patient_data_path, dir)):
            try:
                ecg_time, ecg_signal = load_signal_from_merged_log(patient_data_path + "/" + dir + "/normalized_log.csv", 2)
                ecg_mask = clean_signal(ecg_signal, fs, ecg_clean_config)
                rppg_time, rppg_signal = load_signal_from_merged_log(patient_data_path + "/" + dir + "/normalized_log.csv", 1)
                rppg_mask = clean_signal(rppg_signal, fs, rppg_clean_config)
                accepted, _ = plot_signals(
                    ecg_ax, rppg_ax, ecg_time, ecg_signal, ecg_mask, rppg_signal, rppg_mask,
                    event_handler
                )
                if accepted:
                    try:
                        log_cleaned_data("./cleaned_data/" + dir + ".csv", ecg_time, ecg_signal, rppg_signal, rppg_mask, ecg_mask)
                    except Exception as e:
                        print(f"Error logging cleaned data for {dir}: {e}")
            except Exception as e:
                print(f"Error processing {dir}: {e}")
                continue

    plt.close()

def main():
    fs = 512
    file_path = input("Enter patient data path: ").strip()

    clean_data(file_path, fs)

    # 逐一查看清洗后的数据

    show_cleaned_data("./cleaned_data")

if __name__ == "__main__":
    main()