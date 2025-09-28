from load_data import DataLoader
from patient_info import PatientInfo
import numpy as np
import scipy.signal as signal

def filter_signal(data, fs=512, lowcut=0.5, highcut=5.0, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    data = signal.filtfilt(b, a, data)
    return data

def find_peaks(data, threshold=-0.2, fs=512):
    peaks, _ = signal.find_peaks(data, height=threshold, distance=fs*0.5)
    return peaks

def calculate_hr(peaks, timestamps):
    if len(peaks) < 2:
        return None
    rr_intervals = np.diff(timestamps[peaks])
    hr = 60 / np.mean(rr_intervals) if np.mean(rr_intervals) > 0 else None
    return hr

class PatientStatistics:
    def __init__(self, raw_data_dir, cleaned_data_dir):
        self.raw_data_dir = raw_data_dir
        self.cleaned_data_dir = cleaned_data_dir
        self.patient_list = self.load_all_patients()

    def load_all_patients(self, data_dir="./patient_data", output_file="overall_patient_info.csv"):
        patient_info = PatientInfo(data_dir, save_dir=output_file)
        patient_info_list = patient_info.extract()
        return patient_info_list
    
    def load_data_for_patients(self, patient_list, raw_dir="./patient_data", cleaned_dir="./test_cleaned"):
        patient_ids = [int(p['lab_patient_id']) for p in patient_list]
        data_loader = DataLoader(raw_dir=raw_dir, cleaned_dir=cleaned_dir)
        raw_data = list(data_loader.load_raw_data(patient_id=patient_ids))
        cleaned_data = list(data_loader.load_cleaned_data(patient_id=patient_ids))
        return raw_data, cleaned_data
        
    def hr_stat(self):
        raw_data, cleaned_data = self.load_data_for_patients(self.patient_list)
        rppg_hrs = []
        ecg_hrs = []
        for patient_id, timestamps, rppg_signal, ecg_signal in cleaned_data:
            rppg_signal = filter_signal(rppg_signal, fs=512)
            ecg_signal = filter_signal(ecg_signal, fs=512)
            rppg_peaks = find_peaks(rppg_signal, fs=512)
            ecg_peaks = find_peaks(ecg_signal, fs=512)
            rppg_hr = calculate_hr(rppg_peaks, timestamps)
            ecg_hr = calculate_hr(ecg_peaks, timestamps)
            rppg_hrs.append(rppg_hr)
            ecg_hrs.append(ecg_hr)

        print(f"Average rPPG HR: {np.nanmean(rppg_hrs)}, Average ECG HR: {np.nanmean(ecg_hrs)}")
        print(f"max rPPG HR: {np.nanmax(rppg_hrs)}, max ECG HR: {np.nanmax(ecg_hrs)}")
        print(f"min rPPG HR: {np.nanmin(rppg_hrs)}, min ECG HR: {np.nanmin(ecg_hrs)}")

    def bp_stat(self):
        patient_with_bp = [p for p in self.patient_list if p['low_blood_pressure'] != -1 and p['high_blood_pressure'] != -1]
        low_bps = [p['low_blood_pressure'] for p in patient_with_bp]
        high_bps = [p['high_blood_pressure'] for p in patient_with_bp]
        print(f"Average Low BP: {np.mean(low_bps)}, Average High BP: {np.mean(high_bps)}")
        print(f"Max Low BP: {np.max(low_bps)}, Max High BP: {np.max(high_bps)}")
        print(f"Min Low BP: {np.min(low_bps)}, Min High BP: {np.min(high_bps)}")
    
def main():
    stats = PatientStatistics(raw_data_dir="./patient_data", cleaned_data_dir="./test_cleaned")
    stats.hr_stat()
    stats.bp_stat()

if __name__ == "__main__":
    main()
