import time
import numpy as np
import matplotlib.pyplot as plt

from src.data_loader import get_record_list, load_record
from src.pipeline import process_signal
from src.annotation_loader import load_annotations
from src.window_labeler import label_windows
from src.preprocessing import bandpass_filter, normalize_signal

# -----------------------------
# Config
# -----------------------------
data_path = "data/mitdb"
window_size = 720  # 2-second window
stride = 360       # 50% overlap

# -----------------------------
# 1️⃣ Get all records
# -----------------------------
records = get_record_list(data_path)
print(f"Total records found: {len(records)}")

# -----------------------------
# 2️⃣ Process all records
# -----------------------------
all_windows = []
all_labels = []

for idx, record_name in enumerate(records):
    print(f"\nProcessing record {record_name} ({idx+1}/{len(records)})")
    
    # Load signal
    signal, fs = load_record(data_path, record_name)
    signal = bandpass_filter(signal, fs)
    signal = normalize_signal(signal)

    # Create windows
    start_time = time.time()
    windows = process_signal(signal, fs, window_size=window_size, stride=stride)
    end_time = time.time()
    print(f"Processed windows shape: {windows.shape} | Time: {round(end_time - start_time,4)}s")
    
    # Load annotations & label windows
    samples, symbols = load_annotations(data_path, record_name)
    labels = label_windows(samples, symbols, signal_length=len(signal),
                           window_size=window_size, stride=stride)
    
    labels = np.array(labels, dtype=object)
    
    # Remove None labels
    valid_idx = labels != None
    filtered_windows = windows[valid_idx]
    filtered_labels = labels[valid_idx]
    
    print(f"Valid windows: {len(filtered_windows)} | Normal: {np.sum(filtered_labels==0)} | Abnormal: {np.sum(filtered_labels==1)}")
    
    all_windows.append(filtered_windows)
    all_labels.append(filtered_labels)

# -----------------------------
# 3️⃣ Combine all records
# -----------------------------
all_windows = np.concatenate(all_windows, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

print("\n✅ All records processed")
print(f"Total windows: {all_windows.shape[0]}")
print(f"Normal: {np.sum(all_labels==0)} | Abnormal: {np.sum(all_labels==1)}")

# -----------------------------
# 4️⃣ Save dataset
# -----------------------------
np.save("all_windows.npy", all_windows)
np.save("all_labels.npy", all_labels)
print("Saved all_windows.npy and all_labels.npy")

# -----------------------------
# 5️⃣ Sanity check plots
# -----------------------------
# Pick one normal and one abnormal window
normal_idx = np.where(all_labels==0)[0][0]
abnormal_idx = np.where(all_labels==1)[0][0]

plt.figure(figsize=(10,4))
plt.subplot(2,1,1)
plt.plot(all_windows[normal_idx])
plt.title("Sample Normal Window")

plt.subplot(2,1,2)
plt.plot(all_windows[abnormal_idx])
plt.title("Sample Abnormal Window")
plt.tight_layout()
plt.show()
