import os
import pandas as pd

class DataLoader:
    def __init__(self, raw_dir=None, cleaned_dir=None):
        self.raw_dir = raw_dir
        self.cleaned_dir = cleaned_dir

    def load_raw_data(self, patient_id=[]):
        for f in os.listdir(self.raw_dir):
            if int(f[8:]) in patient_id or not patient_id:
                try:
                    file_path = os.path.join(self.raw_dir, f, "merged_log.csv")
                    df = pd.read_csv(file_path)
                    timestamps = df.iloc[:, 0].astype(float).to_numpy()
                    rppg_signal = df.iloc[:, 1].astype(float).to_numpy()
                    ecg_signal = df.iloc[:, 2].astype(float).to_numpy()
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue

                yield int(f[8:]), timestamps, rppg_signal, ecg_signal

    def load_cleaned_data(self, patient_id=[]):
        for f in os.listdir(self.cleaned_dir):
            if f.endswith(".csv") and (int(f[8:14]) in patient_id or not patient_id):
                try:
                    file_path = os.path.join(self.cleaned_dir, f)
                    df = pd.read_csv(file_path)
                    timestamps = df.iloc[:, 0].astype(float).to_numpy()
                    rppg_signal = df.iloc[:, 1].astype(float).to_numpy()
                    ecg_signal = df.iloc[:, 2].astype(float).to_numpy()
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue

                yield int(f[8:14]), timestamps, rppg_signal, ecg_signal