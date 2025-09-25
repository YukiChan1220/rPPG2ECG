import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import os
import pandas as pd
import threading
from queue import Queue
import global_vars

class DataProcessor:
    def __init__(self, data_path=None, fs=512):
        self.data_path = data_path
        self.fs = fs
        self.time = []
        self.rppg_signal = []
        self.ecg_signal = []
        if data_path is not None:
            self.time, self.rppg_signal, self.ecg_signal = self._load_signal_from_merged_log(data_path)

    def _load_signal_from_merged_log(self, data_path):
        try:
            with open(os.path.join(data_path, 'merged_log.csv'), 'r') as f:
                lines = f.readlines()
                for line in lines[1:]:
                    parts = line.strip().split(',')
                    if len(parts) >= 3:
                        self.time.append(float(parts[0]))
                        self.rppg_signal.append(float(parts[1]) if parts[1] != '' else 0.0)
                        self.ecg_signal.append(float(parts[2]) if parts[2] != '' else 0.0)
            print(f"Loaded {len(self.time)} data points from {data_path}")
            return self.time, self.rppg_signal, self.ecg_signal
        except Exception as e:
            print(f"Error loading {data_path}: {e}")
            return [], [], []

    def _clean_signal(self, sig: list, config: dict):
        sig = np.array(sig)
        mask = np.ones(len(sig), dtype=bool)
        mask[:] = True
        if len(sig) == 0:
            mask[:] = False
            return mask

        if "std" in config:
            window_len = int(config.get("std").get("window_size") * self.fs)
            threshold = config.get("std").get("threshold")
            global_std = np.std(sig)
            for start in range(0, len(sig)-window_len, window_len):
                seg = sig[start:start+window_len]
                seg_std = np.std(seg)
                if seg_std > global_std * threshold:
                    mask[start:start+window_len] = False

        if "diff" in config:
            window_len = int(config.get("diff").get("window_size") * self.fs)
            threshold = config.get("diff").get("threshold")
            global_diff = np.max(sig) - np.min(sig)
            for start in range(0, len(sig)-window_len, window_len):
                seg = sig[start:start+window_len]
                seg_diff = np.max(seg) - np.min(seg)
                if seg_diff > global_diff * threshold:
                    mask[start:start+window_len] = False

        if "welch" in config:
            window_len = int(config.get("welch").get("window_size") * self.fs)
            freq_tolerance = config.get("welch").get("bpm_tolerance") / 60
            gf, gPxx = signal.welch(sig, fs=self.fs, nperseg=window_len)
            peak_freq = gf[np.argmax(gPxx)]
            for start in range(0, len(sig)-window_len, window_len):
                seg = sig[start:start+window_len]
                f, Pxx = signal.welch(seg, fs=self.fs, nperseg=window_len)
                seg_peak_freq = f[np.argmax(Pxx)]
                if abs(seg_peak_freq - peak_freq) > freq_tolerance:
                    mask[start:start+window_len] = False
        return mask
    
    def update_signal(self, signal: str, config: dict):
        if signal == 'rppg':
            return self._clean_signal(self.rppg_signal, config)
        elif signal == 'ecg':
            return self._clean_signal(self.ecg_signal, config)
    
    def set_path(self, data_path):
        self.data_path = data_path
        self.time = []
        self.rppg_signal = []
        self.ecg_signal = []
        self.time, self.rppg_signal, self.ecg_signal = self._load_signal_from_merged_log(data_path)
        
class DataPlotter:
    def __init__(self, data_type: str, plot_event: threading.Event, plot_update_event: threading.Event):
        self.data_type = data_type
        self.time = None
        self.rppg_signal = None
        self.ecg_signal = None
        self.rppg_mask = None
        self.ecg_mask = None
        self.signal_queue = None
        self.config_queue = None
        self.event_queue = None
        self.ecg_config = None
        self.rppg_config = None
        self.fig = None
        self.rppg_ax = None
        self.ecg_ax = None
        self.event = plot_event
        self.plot_update_event = plot_update_event
        self.initialized = False
        if self.data_type == 'raw':
            self._init_raw_plot()
        elif self.data_type == 'cleaned':
            self._init_cleaned_plot()

    def _raw_accept_handler(self, event):
        self.event.set()
        self.event_queue.put('raw_accept')
    def _raw_reject_handler(self, event):
        self.event.set()
        self.event_queue.put('raw_reject')
    def _raw_update_ecg_std_handler(self, val):
        self.ecg_config['std']['threshold'] = val
        self.config_queue.put((self.ecg_config, self.rppg_config))
        self.event.set()
        self.event_queue.put('raw_update')
    def _raw_update_rppg_std_handler(self, val):
        self.rppg_config['std']['threshold'] = val
        self.config_queue.put((self.ecg_config, self.rppg_config))
        self.event.set()
        self.event_queue.put('raw_update')
    def _raw_update_rppg_bpm_handler(self, val):
        self.rppg_config['welch']['bpm_tolerance'] = val
        self.config_queue.put((self.ecg_config, self.rppg_config))
        self.event.set()
        self.event_queue.put('raw_update')

    def _cleaned_accept_handler(self, event):
        self.event.set()
        self.event_queue.put('cleaned_accept')
    def _cleaned_reject_handler(self, event):
        self.event.set()
        self.event_queue.put('cleaned_reject')
    def _cleaned_reverse_handler(self, event):
        self.event.set()
        self.event_queue.put('cleaned_reverse')

    def _init_raw_plot(self):
        plt.ion()
        self.fig, (self.ecg_ax, self.rppg_ax) = plt.subplots(2, 1, figsize=(20, 12))
        plt.subplots_adjust(bottom=0.15)
        ax_accept = plt.axes([0.7, 0.05, 0.1, 0.05])
        ax_reject = plt.axes([0.81, 0.05, 0.1, 0.05])
        ax_ecgstd = plt.axes([0.1, 0.05, 0.1, 0.05])
        ax_rppgstd = plt.axes([0.3, 0.05, 0.1, 0.05])
        ax_rppgbpm = plt.axes([0.5, 0.05, 0.1, 0.05])
        self.ecg_slider = plt.Slider(ax_ecgstd, 'ECG STD Threshold', 0.5, 3.0, valinit=1.5, valstep=0.1)
        self.rppgstd_slider = plt.Slider(ax_rppgstd, 'RPPG STD Threshold', 0.5, 3.0, valinit=1.5, valstep=0.1)
        self.rppgbpm_slider = plt.Slider(ax_rppgbpm, 'RPPG BPM Tolerance', 5, 30, valinit=15, valstep=1)
        self.btn_accept = plt.Button(ax_accept, 'Accept')
        self.btn_reject = plt.Button(ax_reject, 'Reject')
        self.btn_accept.on_clicked(self._raw_accept_handler)
        self.btn_reject.on_clicked(self._raw_reject_handler)
        self.ecg_slider.on_changed(self._raw_update_ecg_std_handler)
        self.rppgstd_slider.on_changed(self._raw_update_rppg_std_handler)
        self.rppgbpm_slider.on_changed(self._raw_update_rppg_bpm_handler)

        self.ecg_config = {
        "std": {
            "window_size": 1,
            "threshold": 1.5
        },
        }
        self.rppg_config = {
            "std": {
                "window_size": 1,
                "threshold": 1.5
            },
            "welch": {
                "window_size": 5,
                "bpm_tolerance": 15
            }
        }

    def _init_cleaned_plot(self):
        plt.ion()
        self.fig, (self.ecg_ax, self.rppg_ax) = plt.subplots(2, 1, figsize=(20, 12))
        plt.subplots_adjust(bottom=0.15)
        ax_accept = plt.axes([0.7, 0.05, 0.1, 0.05])
        ax_reject = plt.axes([0.81, 0.05, 0.1, 0.05])
        ax_reverse = plt.axes([0.59, 0.05, 0.1, 0.05])
        self.btn_accept = plt.Button(ax_accept, 'Accept')
        self.btn_reject = plt.Button(ax_reject, 'Reject')
        self.btn_reverse = plt.Button(ax_reverse, 'Reverse')
        self.btn_accept.on_clicked(self._cleaned_accept_handler)
        self.btn_reject.on_clicked(self._cleaned_reject_handler)
        self.btn_reverse.on_clicked(self._cleaned_reverse_handler)

    def _plot_signals(self):
        self.rppg_ax.clear()
        self.ecg_ax.clear()
        self.rppg_ax.plot(self.time, self.rppg_signal, label='RPPG Signal')
        self.ecg_ax.plot(self.time, self.ecg_signal, label='ECG Signal')
        if self.data_type == 'raw':
            if self.rppg_mask is not None:
                self.rppg_ax.fill_between(self.time, self.rppg_signal, where=~self.rppg_mask, color='red', alpha=0.5, label='Marked Artifacts')
            if self.ecg_mask is not None:
                self.ecg_ax.fill_between(self.time, self.ecg_signal, where=~self.ecg_mask, color='red', alpha=0.5, label='Marked Artifacts')
        self.ecg_ax.set_xlabel('Time')
        self.ecg_ax.set_ylabel('Amplitude')
        self.ecg_ax.set_title('ECG Signal')
        self.ecg_ax.legend()
        self.ecg_ax.grid()
        self.rppg_ax.set_xlabel('Time')
        self.rppg_ax.set_ylabel('Amplitude')
        self.rppg_ax.set_title('RPPG Signal')
        self.rppg_ax.legend()
        self.rppg_ax.grid()
        plt.draw()

    def update_once(self):
        if not self.plot_update_event.is_set():
            return
        try:
            # 非阻塞获取
            from queue import Empty
            item = self.signal_queue.get_nowait()
        except Empty:
            return
        if self.data_type == 'raw':
            self.time, self.rppg_signal, self.ecg_signal, self.rppg_mask, self.ecg_mask = item
        else:
            self.time, self.rppg_signal, self.ecg_signal = item
        self._plot_signals()
        self.plot_update_event.clear()

    def __call__(self, signal_queue: Queue, config_queue: Queue=None, event_queue: Queue=None):
        self.signal_queue = signal_queue
        self.config_queue = config_queue
        self.event_queue = event_queue
        """
        while not global_vars.user_interrupt:
            try:
                self.plot_update_event.wait()
                if self.data_type == 'raw':
                    signal = self.signal_queue.get(timeout=1)
                    self.time, self.rppg_signal, self.ecg_signal, self.rppg_mask, self.ecg_mask = signal
                    self._plot_signals()
                elif self.data_type == 'cleaned':
                    signal = self.signal_queue.get(timeout=1)
                    self.time, self.rppg_signal, self.ecg_signal = signal
                    self._plot_signals()
                self.plot_update_event.clear()
            except Exception as e:
                print(f"Plotting error: {e}")
                import time
                time.sleep(0.1)
            except Exception as e:
                print(f"Plotter error: {e}")
        """

class DataLogger:
    def __init__(self, log_path):
        self.log_path = log_path

    def log_cleaned_data(self, file_path, time, ecg_signal, rppg_signal, rppg_mask, ecg_mask):
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
                with open(os.path.join(self.log_path, file_path.replace('.csv', f'_{file_idx+1}.csv')), 'w') as f:
                    f.write("Time,rPPG,ECG\n")
                    for i in range(start, end):
                        f.write(f"{time[i]},{rppg_signal[i]},{ecg_signal[i]}\n")
                    file_idx += 1

        print(f"Cleaned data logged to {file_path}")

    def modify_cleaned_data(self, file_path, option: str):
        time = []
        rppg_signal = []
        ecg_signal = []
        try:
            if option == 'reject':
                os.remove(os.path.join(self.log_path, file_path))
            elif option == 'reverse' or option == 'accept':
                with open(os.path.join(self.log_path, file_path), 'r') as f:
                    lines = f.readlines()
                    for line in lines[1:]:
                        parts = line.strip().split(',')
                        if len(parts) < 3:
                            continue
                        time.append(float(parts[0]))
                        rppg_signal.append(float(parts[1]) if parts[1] != '' else np.nan)
                        if option == 'accept':
                            ecg_signal.append(float(parts[2]) if parts[2] != '' else np.nan)
                        elif option == 'reverse':
                            ecg_signal.append(-float(parts[2]) if parts[2] != '' else np.nan)
                    # 归一化
                    time = np.array(time)
                    rppg_signal = np.array(rppg_signal)
                    ecg_signal = np.array(ecg_signal)
                    rppg_signal = (rppg_signal - np.mean(rppg_signal)) / np.std(rppg_signal)
                    ecg_signal = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)
                df = pd.DataFrame({
                        'Time': time,
                        'rPPG': rppg_signal,
                        'ECG': ecg_signal
                    })
                df.to_csv(os.path.join(self.log_path, file_path), index=False)
            print(f"Modified {file_path}: {option}")
        except Exception as e:
            print(f"Error {option} {file_path}: {e}")

class Pipeline():
    def __init__(self):
        self.data_processor = None
        self.data_plotter = None
        self.data_logger = None
        self.signal_queue = Queue()
        self.config_queue = Queue()
        self.event_queue = Queue()
        self.plot_event = threading.Event()
        self.update_event = threading.Event()
        self.plot_event.set()

    def start_cleaning(self, data_path, log_path, starting_point=0, ending_point=None):
        self.data_processor = DataProcessor()
        self.data_plotter = DataPlotter('raw', self.plot_event, self.update_event)
        self.data_plotter(self.signal_queue, self.config_queue, self.event_queue)
        self.data_logger = DataLogger(log_path)

        if os.path.exists(log_path) is False:
            os.makedirs(log_path)
        for path in os.listdir(data_path):
            if global_vars.user_interrupt:
                break
            if int(path[8:]) < starting_point or (ending_point is not None and int(path[8:]) > ending_point):
                continue
            self.data_processor.set_path(os.path.join(data_path, path))
            
            initial_rppg_config = {
                "std": {"window_size": 1, "threshold": 1.5},
                "welch": {"window_size": 5, "bpm_tolerance": 15}
            }
            initial_ecg_config = {
                "std": {"window_size": 1, "threshold": 1.5}
            }
            rppg_mask = self.data_processor.update_signal('rppg', initial_rppg_config)
            ecg_mask = self.data_processor.update_signal('ecg', initial_ecg_config)
            self.signal_queue.put((self.data_processor.time, self.data_processor.rppg_signal, self.data_processor.ecg_signal, rppg_mask, ecg_mask))
            self.update_event.set()  # 更新
            
            self.data_plotter.update_once()
            plt.pause(0.01)

            while not global_vars.user_interrupt:
                plt.pause(0.01)
                self.data_plotter.update_once()
                if not self.event_queue.empty():
                    event = self.event_queue.get()
                    if event == 'raw_reject':
                        break
                    elif event == 'raw_update':
                        if not self.config_queue.empty():
                            ecg_config, rppg_config = self.config_queue.get()
                        else:
                            ecg_config = {"std": {"window_size": 1, "threshold": 1.5}}
                            rppg_config = {"std": {"window_size": 1, "threshold": 1.5}, "welch": {"window_size": 5, "bpm_tolerance": 15}}
                        rppg_mask = self.data_processor.update_signal('rppg', rppg_config)
                        ecg_mask = self.data_processor.update_signal('ecg', ecg_config)
                        self.signal_queue.put((self.data_processor.time, self.data_processor.rppg_signal, self.data_processor.ecg_signal, rppg_mask, ecg_mask))
                        self.update_event.set()
                    elif event == 'raw_accept':
                        if not self.config_queue.empty():
                            ecg_config, rppg_config = self.config_queue.get()
                        else:
                            ecg_config = {"std": {"window_size": 1, "threshold": 1.5}}
                            rppg_config = {"std": {"window_size": 1, "threshold": 1.5}, "welch": {"window_size": 5, "bpm_tolerance": 15}}
                        rppg_mask = self.data_processor.update_signal('rppg', rppg_config)
                        ecg_mask = self.data_processor.update_signal('ecg', ecg_config)
                        self.data_logger.log_cleaned_data(f'{path}.csv', self.data_processor.time, self.data_processor.ecg_signal, self.data_processor.rppg_signal, rppg_mask, ecg_mask)
                        break  # 退出内层循环，处理下一个文件
                    else:
                        print(f"Event error: {event}")
        plt.close('all')

    def start_checking_cleaning(self, log_path, starting_point=0, ending_point=None):
        self.data_plotter = DataPlotter('cleaned', self.plot_event, self.update_event)
        self.data_plotter(self.signal_queue, event_queue=self.event_queue)
        self.data_logger = DataLogger(log_path)

        if os.path.exists(log_path) is False:
            print(f"Log path {log_path} does not exist.")
            return
        for file in os.listdir(log_path):
            if global_vars.user_interrupt:
                break
            if not file.endswith('.csv'):
                continue
            if int(file.split('_')[1]) < starting_point or (ending_point is not None and int(file.split('_')[1]) > ending_point):
                continue
            time = []
            rppg_signal = []
            ecg_signal = []
            try:
                with open(os.path.join(log_path, file), 'r') as f:
                    lines = f.readlines()
                    for line in lines[1:]:
                        parts = line.strip().split(',')
                        if len(parts) < 3:
                            continue
                        time.append(float(parts[0]))
                        rppg_signal.append(float(parts[1]) if parts[1] != '' else np.nan)
                        ecg_signal.append(float(parts[2]) if parts[2] != '' else np.nan)
                time = np.array(time)
                rppg_signal = np.array(rppg_signal)
                ecg_signal = np.array(ecg_signal)
                rppg_signal = (rppg_signal - np.mean(rppg_signal)) / np.std(rppg_signal)
                ecg_signal = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)
                self.signal_queue.put((time, rppg_signal, ecg_signal))
                self.update_event.set()
                self.data_plotter.update_once()
                plt.pause(0.01)

                while not global_vars.user_interrupt:
                    plt.pause(0.01)
                    self.data_plotter.update_once()
                    if not self.event_queue.empty():
                        event = self.event_queue.get()
                        if event == 'cleaned_accept':
                            break
                        elif event == 'cleaned_reject':
                            self.data_logger.modify_cleaned_data(file, 'reject')
                            break
                        elif event == 'cleaned_reverse':
                            self.data_logger.modify_cleaned_data(file, 'reverse')
                            break
                        else:
                            print(f"Event error: {event}")
            except Exception as e:
                print(f"Error loading {file}: {e}")
    plt.close('all')

def main():
    data_path = input("Input data path:").strip()
    log_path = input("Input log path:").strip()
    starting_point = input("Input starting point (default 0):").strip()
    ending_point = input("Input ending point (default None):").strip()
    starting_point = int(starting_point) if starting_point.isdigit() else 0
    ending_point = int(ending_point) if ending_point.isdigit() else None
    pipeline = Pipeline()
    pipeline.start_cleaning(data_path=data_path, log_path=log_path, starting_point=starting_point, ending_point=ending_point)
    pipeline.start_checking_cleaning(log_path=log_path, starting_point=starting_point, ending_point=ending_point)

if __name__ == "__main__":
    main()
