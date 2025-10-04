from data.patient_info import PatientInfo
from data.load_data import DataLoader
import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np
import csv

lab = True

if lab:
    data_dir = "./lab_mirror_data"
    output_file = "lab_overall_patient_info.csv"
    merged_patient_file = "lab_merged_patient_info.csv"
    cleaned_dir = "./lab_test_cleaned"
else:
    data_dir = "./patient_data"
    output_file = "overall_patient_info.csv"
    merged_patient_file = "merged_patient_info.csv"
    cleaned_dir = "./test_cleaned"

def load_all_patients(data_dir=data_dir, output_file=output_file):
    patient_info = PatientInfo(data_dir, save_dir=output_file, mode="file")
    patient_info_list = patient_info.extract(data_file=merged_patient_file)
    return patient_info_list

def load_patient_with_bp(data_dir=data_dir, output_file=output_file):
    patient_info = PatientInfo(data_dir, save_dir=output_file, mode="file")
    patient_info_list = patient_info.extract(data_file=merged_patient_file)
    # Convert string values to int for proper comparison
    patient_with_bp = [p for p in patient_info_list if int(p['low_blood_pressure']) != -1 and int(p['high_blood_pressure']) != -1]
    return patient_with_bp

def load_data_for_patients(patient_list, raw_dir=data_dir, cleaned_dir=cleaned_dir):
    patient_ids = [int(p['lab_patient_id']) for p in patient_list]
    data_loader = DataLoader(raw_dir=raw_dir, cleaned_dir=cleaned_dir)
    raw_data = list(data_loader.load_raw_data(patient_id=patient_ids))
    cleaned_data = list(data_loader.load_cleaned_data(patient_id=patient_ids))
    return raw_data, cleaned_data

def filter_signal(data, fs=512, lowcut=0.5, highcut=5.0, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    data = signal.filtfilt(b, a, data)
    return data

def find_peaks(data, threshold=0, fs=512):
    peaks, _ = signal.find_peaks(data, height=threshold, distance=fs*0.5)
    return peaks

def calculate_ptt(rppg_peaks, ecg_peaks, timestamps):
    if abs(len(rppg_peaks) - len(ecg_peaks)) > 1:
        return None, None, None
    min_peak_count = min(len(rppg_peaks), len(ecg_peaks))
    best_align = 0
    best_ptt = 10
    
    for align in range(-2, 3):
        ptt_values = []
        for i in range(min_peak_count):
            rppg_idx = i + align
            ecg_idx = i
            if 0 <= rppg_idx < len(rppg_peaks) and 0 <= ecg_idx < len(ecg_peaks):
                ptt = timestamps[rppg_peaks[rppg_idx]] - timestamps[ecg_peaks[ecg_idx]]
                # ptt should be positive
                if abs(ptt) < 1:
                    ptt_values.append(ptt)
        if len(ptt_values) > 0:
            avg_ptt = np.mean(ptt_values)
            std = np.std(ptt_values)
            if abs(avg_ptt) < abs(best_ptt) and avg_ptt > 0:
                best_ptt = avg_ptt
                best_align = align
    
    return best_ptt, best_align, std

def ptt_signals(timestamps, rppg_signal, ecg_signal, peak=True, filter=True, fs=512):
    if filter:
        rppg_signal = filter_signal(rppg_signal, fs=fs)
        ecg_signal = filter_signal(ecg_signal, fs=fs)
    if peak:
        rppg_peaks = find_peaks(rppg_signal, fs=fs)
        ecg_peaks = find_peaks(ecg_signal, fs=fs)
        ptt, align, std = calculate_ptt(rppg_peaks, ecg_peaks, timestamps)
        if ptt is not None:
            print(f"PTT: {ptt:.3f} seconds, Align: {align}, Std: {std:.3f}")
        return ptt, align, std
    return None, None, None

def plot_signals(timestamps, rppg_signal, ecg_signal, peak=True, filter=True, fs=512):
    if filter:
        filter_signal(rppg_signal, fs=fs)
        filter_signal(ecg_signal, fs=fs)
    if peak:
        rppg_peaks = find_peaks(rppg_signal, fs=fs)
        ecg_peaks = find_peaks(ecg_signal, fs=fs)
        ptt, align, std = calculate_ptt(rppg_peaks, ecg_peaks, timestamps)
        if ptt is not None:
            print(f"PTT: {ptt:.3f} seconds, Align: {align}, Std: {std:.3f}")

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
    option = input(f"STD: {std}, Y to accept, N to reject: ")
    if option.lower() == 'n':
        return None, None, None
    return ptt, align, std


def show_patient_with_hr():
    patient_list = load_patient_with_bp()
    raw_data, cleaned_data = load_data_for_patients(patient_list)

    for patient_id, timestamps, rppg_signal, ecg_signal in cleaned_data:
        print(f"Patient ID: {patient_id}")
        plot_signals(timestamps, rppg_signal, ecg_signal, peak=True, filter=True, fs=512)

bp_patient_list = load_patient_with_bp()
raw_data, cleaned_data = load_data_for_patients(bp_patient_list)

bp_dict = {int(p['lab_patient_id']): (int(p['low_blood_pressure']), int(p['high_blood_pressure'])) 
           for p in bp_patient_list}

ptt_list = []
for patient_id, timestamps, rppg_signal, ecg_signal in cleaned_data:
    print(f"Patient ID: {patient_id}")
    #ptt, align, std = plot_signals(timestamps, rppg_signal, ecg_signal, peak=True, filter=True, fs=512)
    ptt, align, std = ptt_signals(timestamps, rppg_signal, ecg_signal, peak=True, filter=True, fs=512)
    ptt_list.append((patient_id, ptt, align, std))

best_coef = 0
best_std = 0
best_ptt_low_threshold = 0
for std_threshold in np.linspace(0.005, 0.2, 40):
    for ptt_low_threshold in np.linspace(0, 0.2, 21):
        ptt_bp = []
        for patient_id, ptt, align, std in ptt_list:
            if std is None or std > std_threshold:
                continue
            if patient_id in bp_dict:
                low_bp, high_bp = bp_dict[patient_id]
                if(ptt != None and ptt_low_threshold < abs(ptt) and abs(ptt) < 1 and low_bp != -1 and high_bp != -1):
                    ptt_bp.append((patient_id, ptt, low_bp, high_bp, std))

        # calculate the average ptt for patients
        ptt_dict = {}
        for patient_id, ptt, low_bp, high_bp, std in ptt_bp:
            if ptt is not None:
                if patient_id not in ptt_dict:
                    ptt_dict[patient_id] = []
                ptt_dict[patient_id].append((ptt, low_bp, high_bp, std))
        ptt_bp = [(np.mean([p[0] for p in v]), v[0][1], v[0][2], np.mean([p[3] for p in v])) for k, v in ptt_dict.items()]
        ptt_bp.sort(key=lambda x: x[3])  # sort by std

        ptt_values = []
        low_bps = []
        high_bps = []
        mean_bps = []
        for i in range(len(ptt_bp)):
            ptt_values.append(ptt_bp[i][0])
            high_bps.append(ptt_bp[i][2])
            low_bps.append(ptt_bp[i][1])
            mean_bps.append((ptt_bp[i][1]+ptt_bp[i][2])/2)

        if len(ptt_values) < 10:
            continue

        ptt_values = np.array(ptt_values)
        ptt_values_rec = np.reciprocal(ptt_values)
        low_bps = np.array(low_bps)
        high_bps = np.array(high_bps)
        mean_bps = np.array(mean_bps)

        low_coef = np.corrcoef(ptt_values, low_bps)[0, 1]
        low_coef_rec = np.corrcoef(ptt_values_rec, low_bps)[0, 1]
        high_coef = np.corrcoef(ptt_values, high_bps)[0, 1]
        high_coef_rec = np.corrcoef(ptt_values_rec, high_bps)[0, 1]
        mean_coef = np.corrcoef(ptt_values, mean_bps)[0, 1]
        mean_coef_rec = np.corrcoef(ptt_values_rec, mean_bps)[0, 1]
        coef = low_coef + high_coef + mean_coef
        if coef < best_coef and low_coef < 0 and high_coef < 0 and mean_coef < 0:
            best_coef = coef
            best_std = std_threshold
            best_ptt_low_threshold = ptt_low_threshold
            print(f"New best coef: {best_coef:.2f} with std threshold: {best_std}, ptt low threshold: {best_ptt_low_threshold}")
            print(f"Low coef: {low_coef:.2f}, High coef: {high_coef:.2f}, Mean coef: {mean_coef:.2f}")
            print(f"Low coef rec: {low_coef_rec:.2f}, High coef rec: {high_coef_rec:.2f}, Mean coef rec: {mean_coef_rec:.2f}")
print(f"Best coef: {best_coef:.2f} with std threshold: {best_std}, ptt low threshold: {best_ptt_low_threshold}")

ptt_bp = []
for patient_id, ptt, align, std in ptt_list:
    if std is None or std > 0.035:
        continue
    if patient_id in bp_dict:
        low_bp, high_bp = bp_dict[patient_id]
        if(ptt != None and 0.11 < abs(ptt) and abs(ptt) < 1 and low_bp != -1 and high_bp != -1):
            ptt_bp.append((patient_id, ptt, low_bp, high_bp, std))
            print(f"Patient ID: {patient_id}, PTT: {ptt:.3f} seconds, Low BP: {low_bp}, High BP: {high_bp}, Std: {std:.3f}")

# calculate the average ptt for patients
ptt_dict = {}
for patient_id, ptt, low_bp, high_bp, std in ptt_bp:
    if ptt is not None:
        if patient_id not in ptt_dict:
            ptt_dict[patient_id] = []
        ptt_dict[patient_id].append((ptt, low_bp, high_bp, std))
ptt_bp = [(np.mean([p[0] for p in v]), v[0][1], v[0][2], np.mean([p[3] for p in v])) for k, v in ptt_dict.items()]
ptt_bp.sort(key=lambda x: x[3])  # sort by std

ptt_values = []
low_bps = []
high_bps = []
mean_bps = []
for i in range(len(ptt_bp)):
    ptt_values.append(ptt_bp[i][0])
    high_bps.append(ptt_bp[i][2])
    low_bps.append(ptt_bp[i][1])
    mean_bps.append((ptt_bp[i][1]+ptt_bp[i][2])/2)

ptt_values = np.array(ptt_values)
ptt_values_rec = np.reciprocal(ptt_values)
low_bps = np.array(low_bps)
high_bps = np.array(high_bps)
mean_bps = np.array(mean_bps)

low_coef = np.corrcoef(ptt_values, low_bps)[0, 1]
low_coef_rec = np.corrcoef(ptt_values_rec, low_bps)[0, 1]
high_coef = np.corrcoef(ptt_values, high_bps)[0, 1]
high_coef_rec = np.corrcoef(ptt_values_rec, high_bps)[0, 1]
mean_coef = np.corrcoef(ptt_values, mean_bps)[0, 1]
mean_coef_rec = np.corrcoef(ptt_values_rec, mean_bps)[0, 1]

print("Correlation Coefficients:")
print(f"PTT - DBP: {low_coef:.2f}")
print(f"PTT - SBP: {high_coef:.2f}")
print(f"PTT - DBP-SBP Mean: {mean_coef:.2f}")
print(f"1/PTT - DBP: {low_coef_rec:.2f}")
print(f"1/PTT - SBP: {high_coef_rec:.2f}")
print(f"1/PTT - DBP-SBP Mean: {mean_coef_rec:.2f}")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
low_bp_ax, high_bp_ax, mean_bp_ax = axes[0]
low_bp_rec_ax, high_bp_rec_ax, mean_bp_rec_ax = axes[1]

ptt_maxmin = [min(ptt_values), max(ptt_values)]

low_bp_ax.scatter(ptt_values, low_bps, label='DBP', color='blue', alpha=0.7)
low_bp_ax.plot(ptt_maxmin, np.poly1d(np.polyfit(ptt_values, low_bps, 1))(ptt_maxmin), color='orange', linestyle='--', label='Fit Line')
low_bp_rec_ax.scatter(ptt_values_rec, low_bps, label='DBP', color='blue', alpha=0.7)
high_bp_ax.scatter(ptt_values, high_bps, label='SBP', color='red', alpha=0.7)
high_bp_ax.plot(ptt_maxmin, np.poly1d(np.polyfit(ptt_values, high_bps, 1))(ptt_maxmin), color='orange', linestyle='--', label='Fit Line')
high_bp_rec_ax.scatter(ptt_values_rec, high_bps, label='SBP', color='red', alpha=0.7)
mean_bp_ax.scatter(ptt_values, mean_bps, label='DBP-SBP Mean', color='green', alpha=0.7)
mean_bp_ax.plot(ptt_maxmin, np.poly1d(np.polyfit(ptt_values, mean_bps, 1))(ptt_maxmin), color='orange', linestyle='--', label='Fit Line')
mean_bp_rec_ax.scatter(ptt_values_rec, mean_bps, label='DBP-SBP Mean', color='green', alpha=0.7)

high_bp_ax.set_title(f'PTT - SBP')
high_bp_ax.set_xlabel('PTT (seconds)')
high_bp_ax.set_ylabel('Blood Pressure (mmHg)')
high_bp_ax.legend()
high_bp_ax.grid()

low_bp_ax.set_title(f'PTT - DBP')
low_bp_ax.set_xlabel('PTT (seconds)')
low_bp_ax.set_ylabel('Blood Pressure (mmHg)')
low_bp_ax.legend()
low_bp_ax.grid()

mean_bp_ax.set_title(f'PTT - DBP-SBP Mean')
mean_bp_ax.set_xlabel('PTT (seconds)')
mean_bp_ax.set_ylabel('Blood Pressure (mmHg)')
mean_bp_ax.legend()
mean_bp_ax.grid()

high_bp_rec_ax.set_title(f'1/PTT - SBP')
high_bp_rec_ax.set_xlabel('1/PTT (1/seconds)')
high_bp_rec_ax.set_ylabel('Blood Pressure (mmHg)')
high_bp_rec_ax.legend()
high_bp_rec_ax.grid()

low_bp_rec_ax.set_title(f'1/PTT - DBP')
low_bp_rec_ax.set_xlabel('1/PTT (1/seconds)')
low_bp_rec_ax.set_ylabel('Blood Pressure (mmHg)')
low_bp_rec_ax.legend()
low_bp_rec_ax.grid()

mean_bp_rec_ax.set_title(f'1/PTT - DBP-SBP Mean')
mean_bp_rec_ax.set_xlabel('1/PTT (1/seconds)')
mean_bp_rec_ax.set_ylabel('Blood Pressure (mmHg)')
mean_bp_rec_ax.legend()
mean_bp_rec_ax.grid()

plt.show()

# write ptt and bp to csv

with open('ptt_bp.csv', 'w', newline='') as csvfile:
    fieldnames = ['ptt', 'low_blood_pressure', 'high_blood_pressure']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for ptt, low_bp, high_bp in ptt_bp:
        if ptt is not None:
            writer.writerow({'ptt': ptt, 'low_blood_pressure': low_bp, 'high_blood_pressure': high_bp})


