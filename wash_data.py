import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import os

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


def plot_signals(ecg_ax, rppg_ax, time, ecg_signal, ecg_mask, rppg_signal, rppg_mask, ecg_event_handler, rppg_event_handler):
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
    while not ecg_event_handler.status or not rppg_event_handler.status:
        plt.pause(0.1)
    ecg_ax.cla()
    rppg_ax.cla()
    ecg_event_handler.status = False
    rppg_event_handler.status = False
    return ecg_event_handler.accept, rppg_event_handler.accept


class PltEventHandler:
    def __init__(self):
        self.status = False
        self.accept = False

    def accept_handler(self):
        def handler(event):
            self.status = True
            self.accept = True
        return handler

    def reject_handler(self):
        def handler(event):
            self.status = True
            self.accept = False
        return handler

def main():
    file_path = None
    fs = 512
    file_path = input("Enter patient data path: ").strip()

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
    plt.subplots_adjust(bottom=0.25)
    ecg_event_handler = PltEventHandler()
    rppg_event_handler = PltEventHandler()
    ecgaccept = plt.axes([0.7, 0.15, 0.1, 0.05])
    ecgreject = plt.axes([0.81, 0.15, 0.1, 0.05])
    rppgaccept = plt.axes([0.7, 0.05, 0.1, 0.05])
    rppgreject = plt.axes([0.81, 0.05, 0.1, 0.05])
    becgaccept = plt.Button(ecgaccept, 'Accept')
    becgreject = plt.Button(ecgreject, 'Reject')
    brppgaccept = plt.Button(rppgaccept, 'Accept')
    brppgreject = plt.Button(rppgreject, 'Reject')
    becgaccept.on_clicked(ecg_event_handler.accept_handler())
    becgreject.on_clicked(ecg_event_handler.reject_handler())
    brppgaccept.on_clicked(rppg_event_handler.accept_handler())
    brppgreject.on_clicked(rppg_event_handler.reject_handler())


    for dir in os.listdir(file_path):
        if os.path.isdir(os.path.join(file_path, dir)):
            ecg_time, ecg_signal = load_signal_from_merged_log(file_path + "/" + dir + "/normalized_log.csv", 2)
            ecg_mask = clean_signal(ecg_signal, fs, ecg_clean_config)
            rppg_time, rppg_signal = load_signal_from_merged_log(file_path + "/" + dir + "/normalized_log.csv", 1)
            rppg_mask = clean_signal(rppg_signal, fs, rppg_clean_config)
            ecg_accepted, rppg_accepted = plot_signals(
                ecg_ax, rppg_ax, ecg_time, ecg_signal, ecg_mask, rppg_signal, rppg_mask,
                ecg_event_handler, rppg_event_handler
            )


if __name__ == "__main__":
    main()