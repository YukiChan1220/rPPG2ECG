from data.patient_info import PatientInfo
from data.load_data import DataLoader
import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np

def load_all_patients(data_dir="./patient_data", output_file="overall_patient_info.csv"):
    patient_info = PatientInfo(data_dir, save_dir=output_file)
    patient_info_list = patient_info.extract()
    return patient_info_list

def load_patient_with_bp(data_dir="./patient_data", output_file="overall_patient_info.csv"):
    patient_info = PatientInfo(data_dir, save_dir=output_file)
    patient_info_list = patient_info.extract()
    patient_with_bp = [p for p in patient_info_list if p['low_blood_pressure'] != -1 and p['high_blood_pressure'] != -1]
    return patient_with_bp

def load_data_for_patients(patient_list, raw_dir="./patient_data", cleaned_dir="./test_cleaned"):
    patient_ids = [int(p['lab_patient_id']) for p in patient_list]
    data_loader = DataLoader(raw_dir=raw_dir, cleaned_dir=cleaned_dir)
    raw_data = list(data_loader.load_raw_data(patient_id=patient_ids))
    cleaned_data = list(data_loader.load_cleaned_data(patient_id=patient_ids))
    return raw_data, cleaned_data

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

def calculate_ptt(rppg_peaks, ecg_peaks, timestamps):
    if abs(len(rppg_peaks) - len(ecg_peaks)) > 2:
        return None
    align = 0
    best_ptt = None
    for align in range(-5, 5):
        if len(rppg_peaks) + align > 0 and len(ecg_peaks) > 0:
            align = align
            break
    return best_ptt

def plot_signals(timestamps, rppg_signal, ecg_signal, peak=True, filter=True, fs=512):
    if filter:
        filter_signal(rppg_signal, fs=fs)
        filter_signal(ecg_signal, fs=fs)
    if peak:
        rppg_peaks = find_peaks(rppg_signal, fs=fs)
        ecg_peaks = find_peaks(ecg_signal, fs=fs)

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(timestamps, rppg_signal, label='rPPG Signal')
    if peak:
        plt.plot(timestamps[rppg_peaks], rppg_signal[rppg_peaks], "x", label='rPPG Peaks')
    plt.title('rPPG Signal with Peaks')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(timestamps, ecg_signal, label='ECG Signal')
    if peak:
        plt.plot(timestamps[rppg_peaks], rppg_signal[rppg_peaks], "x", label='rPPG Peaks')
        plt.plot(timestamps[ecg_peaks], ecg_signal[ecg_peaks], "o", label='ECG Peaks')
    plt.title('ECG Signal with Peaks')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.show()


def show_patient_with_hr():
    patient_list = load_patient_with_bp()
    raw_data, cleaned_data = load_data_for_patients(patient_list)

    for patient_id, timestamps, rppg_signal, ecg_signal in cleaned_data:
        print(f"Patient ID: {patient_id}")
        plot_signals(timestamps, rppg_signal, ecg_signal, peak=True, filter=True, fs=512)

def calculate_all_hr():
    patient_list = load_all_patients()
    raw_data, cleaned_data = load_data_for_patients(patient_list)
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


calculate_all_hr()
