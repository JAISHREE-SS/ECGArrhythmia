'''
import wfdb
import matplotlib.pyplot as plt

# Load record 100 from MIT-BIH
record = wfdb.rdrecord('data/mitdb/100')

# Extract MLII lead (channel 0)
signal = record.p_signal[:, 0]

# Get sampling frequency
fs = record.fs

print("Sampling rate:", fs)

# Plot first 5 seconds
plt.figure(figsize=(12, 4))
plt.plot(signal[:fs * 5])
plt.title("Raw ECG - Record 100 (First 5 Seconds)")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
print("Signal length:", len(signal))
print("Duration (seconds):", len(signal)/fs)
'''
'''
import wfdb
import matplotlib.pyplot as plt
from src.preprocessing import bandpass_filter, normalize_signal

record = wfdb.rdrecord('data/mitdb/100')
signal = record.p_signal[:, 0]
fs = record.fs

filtered = bandpass_filter(signal, fs)
normalized = normalize_signal(filtered)

plt.figure(figsize=(12,6))

plt.subplot(2,1,1)
plt.plot(signal[:fs*5])
plt.title("Raw ECG")

plt.subplot(2,1,2)
plt.plot(normalized[:fs*5])
plt.title("Filtered + Normalized ECG")

plt.tight_layout()
plt.show()
'''
'''
import wfdb
import matplotlib.pyplot as plt
from src.preprocessing import bandpass_filter, normalize_signal
from src.sliding_window import create_windows
from src.realtime_sim import realtime_stream
from src.pipeline import process_signal


# Load record
record = wfdb.rdrecord('data/mitdb/100')
signal = record.p_signal[:, 0]
fs = record.fs

# Preprocess
filtered = bandpass_filter(signal, fs)
normalized = normalize_signal(filtered)

# Create sliding windows
windows = process_signal(signal, fs)
print("Processed windows shape:", windows.shape)

print("Total windows:", len(windows))
print("Shape of one window:", windows[0].shape)

# Plot first 5 seconds raw vs processed
plt.figure(figsize=(12,6))

plt.subplot(2,1,1)
plt.plot(signal[:fs*5])
plt.title("Raw ECG")

plt.subplot(2,1,2)
plt.plot(normalized[:fs*5])
plt.title("Filtered + Normalized ECG")

plt.tight_layout()
plt.show()
print("\nStarting Real-Time Simulation...\n")

for window in realtime_stream(normalized, fs, delay=0.5):
    print("Window received:", window.shape)
    break  # remove this later, just testing first window

import time

start = time.time()
windows = create_windows(normalized, 900, 360)
end = time.time()

print("Window creation time:", end - start, "seconds")
'''
'''
import time
import matplotlib.pyplot as plt
from src.data_loader import get_record_list, load_record
from src.pipeline import process_signal
from src.realtime_sim import realtime_stream


data_path = "data/mitdb"

# -----------------------------
# 1️⃣ Check Available Records
# -----------------------------
records = get_record_list(data_path)
print("Total records found:", len(records))


# -----------------------------
# 2️⃣ Process First Record Fully
# -----------------------------
record_name = records[0]
print("\nProcessing record:", record_name)

signal, fs = load_record(data_path, record_name)

start = time.time()
windows = process_signal(signal, fs)
end = time.time()

print("Processed windows shape:", windows.shape)
print("Window creation time:", end - start, "seconds")


# -----------------------------
# 3️⃣ Plot Raw vs Processed
# -----------------------------
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(signal[:fs*5])
plt.title("Raw ECG")

plt.subplot(2, 1, 2)
plt.plot(windows[0])  # First processed window
plt.title("First Processed Window (900 samples)")

plt.tight_layout()
plt.show()


# -----------------------------
# 4️⃣ Real-Time Simulation Test
# -----------------------------
print("\nStarting Real-Time Simulation...\n")

for window in realtime_stream(signal, fs, delay=0.5):
    print("Window received:", window.shape)
    break  # only test first window


# -----------------------------
# 5️⃣ Test Multiple Records (Quick Check)
# -----------------------------
print("\nTesting first 3 records for stability...\n")

for record_name in records[:3]:
    print("Processing:", record_name)
    signal, fs = load_record(data_path, record_name)
    windows = process_signal(signal, fs)
    print("Windows shape:", windows.shape)

from src.annotation_loader import load_annotations

samples, symbols = load_annotations(data_path, record_name)

print("First 10 annotation samples:", samples[:10])
print("First 10 symbols:", symbols[:10])
print("Unique symbols in record:", set(symbols))

from src.annotation_loader import load_annotations
from src.window_labeler import label_windows

samples, symbols = load_annotations(data_path, record_name)

labels = label_windows(
    samples,
    symbols,
    signal_length=len(signal)
)

print("Total window labels:", len(labels))
print("First 20 labels:", labels[:20])

print("Normal windows:", sum(labels == 0))
print("Abnormal windows:", sum(labels == 1))
print("None windows:", sum(labels == None))

valid_indices = labels != None

filtered_windows = windows[valid_indices]
filtered_labels = labels[valid_indices]

print("\nAfter removing None windows:")
print("Final training windows:", len(filtered_windows))
print("Normal:", sum(filtered_labels == 0))
print("Abnormal:", sum(filtered_labels == 1))
'''
'''
import time
import matplotlib.pyplot as plt
import numpy as np

from src.data_loader import get_record_list, load_record
from src.pipeline import process_signal
from src.realtime_sim import realtime_stream
from src.annotation_loader import load_annotations
from src.window_labeler import label_windows


data_path = "data/mitdb"

# -----------------------------
# 1️⃣ Check Available Records
# -----------------------------
records = get_record_list(data_path)
print("Total records found:", len(records))


# -----------------------------
# 2️⃣ Process First Record Fully
# -----------------------------
record_name = records[0]
print("\nProcessing record:", record_name)

signal, fs = load_record(data_path, record_name)

start = time.time()
windows = process_signal(signal, fs)   # Make sure this uses 432, 180
end = time.time()

print("Processed windows shape:", windows.shape)
print("Window creation time:", end - start, "seconds")


# -----------------------------
# 3️⃣ Plot Raw vs First Window
# -----------------------------
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(signal[:fs*5])
plt.title("Raw ECG (First 5 Seconds)")

plt.subplot(2, 1, 2)
plt.plot(windows[0])
plt.title("First Processed Window (432 samples)")

plt.tight_layout()
plt.show()


# -----------------------------
# 4️⃣ Real-Time Simulation Test
# -----------------------------
print("\nStarting Real-Time Simulation...\n")

for window in realtime_stream(signal, fs, delay=0.5):
    print("Window received:", window.shape)
    break


# -----------------------------
# 5️⃣ Annotation + Labeling
# -----------------------------
samples, symbols = load_annotations(data_path, record_name)

print("\nFirst 10 annotation samples:", samples[:10])
print("First 10 symbols:", symbols[:10])
print("Unique symbols in record:", set(symbols))

labels = label_windows(
    samples,
    symbols,
    signal_length=len(signal),
    window_size=432,
    stride=180
)

print("\nTotal window labels:", len(labels))
print("First 20 labels:", labels[:20])

print("Normal windows:", np.sum(labels == 0))
print("Abnormal windows:", np.sum(labels == 1))
print("None windows:", np.sum(labels == None))


# -----------------------------
# 6️⃣ Remove None Windows
# -----------------------------
valid_indices = labels != None

filtered_windows = windows[valid_indices]
filtered_labels = labels[valid_indices]

print("\nAfter removing None windows:")
print("Final training windows:", len(filtered_windows))
print("Normal:", np.sum(filtered_labels == 0))
print("Abnormal:", np.sum(filtered_labels == 1))


# -----------------------------
# 7️⃣ Stability Test (3 Records)
# -----------------------------
print("\nTesting first 3 records for stability...\n")

for record_name in records[:3]:
    print("Processing:", record_name)
    signal, fs = load_record(data_path, record_name)
    windows = process_signal(signal, fs)
    print("Windows shape:", windows.shape)
'''
'''
import time
import matplotlib.pyplot as plt
import numpy as np

from src.data_loader import get_record_list, load_record
from src.pipeline import process_signal
from src.realtime_sim import realtime_stream
from src.annotation_loader import load_annotations
from src.window_labeler import label_windows

# -----------------------------
# Config
# -----------------------------
data_path = "data/mitdb"
window_size = 720
stride = 360

# -----------------------------
# 1️⃣ Get All Records
# -----------------------------
records = get_record_list(data_path)
print("Total records found:", len(records))

# -----------------------------
# 2️⃣ Process All Records
# -----------------------------
all_windows = []
all_labels = []

for idx, record_name in enumerate(records):
    print(f"\nProcessing record {record_name} ({idx+1}/{len(records)})")
    
    # Load signal
    signal, fs = load_record(data_path, record_name)
    
    # Process windows
    start_time = time.time()
    windows = process_signal(signal, fs)
    end_time = time.time()
    print("Processed windows shape:", windows.shape)
    print("Window creation time:", round(end_time - start_time, 4), "seconds")
    
    # Load annotations
    samples, symbols = load_annotations(data_path, record_name)
    
    # Label windows
    labels = label_windows(
        samples,
        symbols,
        signal_length=len(signal),
        window_size=window_size,
        stride=stride
    )
    
    labels = np.array(labels, dtype=object)  # safe comparison
    
    # Remove None labels
    valid_idx = labels != None
    filtered_windows = windows[valid_idx]
    filtered_labels = labels[valid_idx]
    
    print(f"Valid windows: {len(filtered_windows)} | Normal: {np.sum(filtered_labels==0)} | Abnormal: {np.sum(filtered_labels==1)}")
    
    # Append to master list
    all_windows.append(filtered_windows)
    all_labels.append(filtered_labels)

# Combine all records
all_windows = np.concatenate(all_windows, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

print("\n✅ All records processed")
print("Total windows:", all_windows.shape[0])
print("Normal windows:", np.sum(all_labels==0))
print("Abnormal windows:", np.sum(all_labels==1))

# -----------------------------
# 3️⃣ Plot Raw vs First Window of First Record
# -----------------------------
signal, fs = load_record(data_path, records[0])
windows = process_signal(signal, fs)

plt.figure(figsize=(12, 6))

plt.subplot(2,1,1)
plt.plot(signal[:720])
plt.title("Raw ECG (First 2 Seconds)")

plt.subplot(2,1,2)
plt.plot(windows[0])
plt.title(f"First Processed Window ({window_size} samples)")

plt.tight_layout()
plt.show()

# -----------------------------
# 4️⃣ Real-Time Simulation for First Record
# -----------------------------
print("\nStarting Real-Time Simulation (first record only)...\n")
for i, window in enumerate(realtime_stream(signal, fs, delay=0.5)):
    print(f"Streaming window {i+1}/{windows.shape[0]} | Shape: {window.shape}")
    if i == 9:  # limit to first 10 windows for testing
        break

# -----------------------------
# 5️⃣ Stability Test on First 3 Records
# -----------------------------
print("\nTesting stability on first 3 records...\n")
for record_name in records[:3]:
    signal, fs = load_record(data_path, record_name)
    windows = process_signal(signal, fs)
    print(f"Record {record_name} | Windows shape: {windows.shape}")
'''
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1️⃣ Load saved dataset
# -----------------------------
all_windows = np.load("all_windows.npy", allow_pickle=True)
all_labels = np.load("all_labels.npy", allow_pickle=True)

print("✅ Dataset loaded")
print("Total windows:", all_windows.shape[0])
print("Normal:", np.sum(all_labels==0))
print("Abnormal:", np.sum(all_labels==1))

# -----------------------------
# 2️⃣ Sanity check plots
# -----------------------------
# Plot first 5 normal and abnormal windows
normal_idx = np.where(all_labels==0)[0][:5]
abnormal_idx = np.where(all_labels==1)[0][:5]

plt.figure(figsize=(12,6))
plt.subplot(2,1,1)
plt.plot(all_windows[normal_idx[0]].squeeze())
plt.title("Sample Normal Window")

plt.subplot(2,1,2)
plt.plot(all_windows[abnormal_idx[0]].squeeze())
plt.title("Sample Abnormal Window")

plt.tight_layout()
plt.show()

# -----------------------------
# 3️⃣ Quick stats
# -----------------------------
print("First 5 labels:", all_labels[:5])
print("Shape of a window:", all_windows[0].shape)
