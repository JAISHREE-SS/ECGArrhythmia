'''
import time
import matplotlib.pyplot as plt
import numpy as np
from src.data_loader import load_record
from src.pipeline import process_signal
from src.realtime_sim import realtime_stream

# -----------------------------
# Config
# -----------------------------
record_name = "100"       # choose first record for now
window_size = 432
stride = 180
delay = 0.1               # simulate real-time streaming

# -----------------------------
# Load signal and windows
# -----------------------------
signal, fs = load_record("data/mitdb", record_name)
windows = process_signal(signal, fs)
num_windows = windows.shape[0]

# -----------------------------
# Helper: Dummy Alert System
# -----------------------------
def dummy_alert(window_idx):
    if window_idx % 10 == 0:  # every 10th window is "abnormal"
        return True
    return False

# -----------------------------
# Helper: Noise Injection
# -----------------------------
def add_noise(window, sigma=0.02):
    return window + np.random.normal(0, sigma, size=window.shape)

# -----------------------------
# Interactive Plot Setup
# -----------------------------
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(windows[0])
#ax.set_ylim(np.min(windows), np.max(windows))
ax.set_ylim(-1, 2)  
ax.set_title("Live ECG Streaming (Dummy Alerts)")

# -----------------------------
# Real-Time Streaming Loop
# -----------------------------
for i, window in enumerate(realtime_stream(signal, fs, delay=delay)):
    noisy_window = add_noise(window)
    line.set_ydata(noisy_window)
    
    # Dummy alert check
    if dummy_alert(i):
        ax.set_facecolor("mistyrose")
        print(f"ALERT: Abnormal rhythm detected in window {i+1}")
    else:
        ax.set_facecolor("white")
    
    ax.set_title(f"Window {i+1}/{num_windows}")
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    if i == 50:  # stop after 50 windows for testing
        break

plt.ioff()
plt.show()

# -----------------------------
# Comparison Mode Example
# -----------------------------
plt.figure(figsize=(10,4))
plt.subplot(2,1,1)
plt.plot(windows[10])
plt.title("Abnormal Window (Dummy)")
plt.subplot(2,1,2)
plt.plot(windows[0])
plt.title("Normal Reference Window")
plt.tight_layout()
plt.show()
'''
'''
import time
import matplotlib.pyplot as plt
import numpy as np
from src.data_loader import load_record
from src.pipeline import process_signal
from src.realtime_sim import realtime_stream
from src.preprocessing import bandpass_filter, normalize_signal

# -----------------------------
# Config
# -----------------------------
record_name = "100"   # Local MIT-BIH record
data_path = "data/mitdb"
window_size = 432
stride = 180
delay = 0.1           # seconds per window
show_comparison = True

# -----------------------------
# Load and preprocess signal
# -----------------------------
signal, fs = load_record(data_path, record_name)
signal = bandpass_filter(signal, fs)
signal = normalize_signal(signal)

windows = process_signal(signal, fs, window_size=window_size, stride=stride)
num_windows = windows.shape[0]

# -----------------------------
# Dummy prediction/alert function
# -----------------------------
def dummy_prediction(window_idx):
    """Simulate prediction confidence"""
    if window_idx % 10 == 0:
        return np.random.uniform(0.7, 1.0)  # abnormal
    return np.random.uniform(0.0, 0.3)      # normal

def check_alert(conf):
    return conf > 0.5

# -----------------------------
# Setup plot
# -----------------------------
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(windows[0])
ax.set_ylim(np.min(windows)-0.1, np.max(windows)+0.1)
ax.set_title("Live ECG Dashboard (Local Record)")

# -----------------------------
# Real-time streaming loop
# -----------------------------
for i, window in enumerate(realtime_stream(signal, fs, delay=delay)):
    # Optional: add very small noise for realism
    noisy_window = window + np.random.normal(0, 0.005, size=window.shape)

    line.set_ydata(noisy_window)

    conf = dummy_prediction(i)
    if check_alert(conf):
        ax.set_facecolor("mistyrose")
        print(f"ALERT: Abnormal detected in window {i+1} | Confidence: {conf:.2f}")
    else:
        ax.set_facecolor("white")

    ax.set_title(f"Window {i+1}/{num_windows} | Confidence: {conf:.2f}")
    fig.canvas.draw()
    fig.canvas.flush_events()

    if i == 50:  # limit for testing
        break

plt.ioff()
plt.show()

# -----------------------------
# Optional Comparison Mode
# -----------------------------
if show_comparison:
    plt.figure(figsize=(10,4))
    abnormal_idx = 10
    normal_idx = 0
    plt.subplot(2,1,1)
    plt.plot(windows[abnormal_idx])
    plt.title("Abnormal Window (Dummy)")

    plt.subplot(2,1,2)
    plt.plot(windows[normal_idx])
    plt.title("Normal Reference Window")
    plt.tight_layout()
    plt.show()
'''
'''
import time
import matplotlib.pyplot as plt
import numpy as np
from src.data_loader import load_record
from src.pipeline import process_signal
from src.realtime_sim import realtime_stream
from src.preprocessing import bandpass_filter, normalize_signal

# -----------------------------
# Config
# -----------------------------
record_name = "100"
data_path = "data/mitdb"
window_size = 720
stride = 360
human_hr_factor = 0.02  # ~50 bpm scaling for visualization
show_comparison = True

# -----------------------------
# Load and preprocess
# -----------------------------
signal, fs = load_record(data_path, record_name)
signal = bandpass_filter(signal, fs)
signal = normalize_signal(signal)
windows = process_signal(signal, fs, window_size=window_size, stride=stride)
num_windows = windows.shape[0]

# -----------------------------
# Dummy prediction / alert
# -----------------------------
def dummy_prediction(window_idx):
    if window_idx % 10 == 0:
        return np.random.uniform(0.7, 1.0)
    return np.random.uniform(0.0, 0.3)

def check_alert(conf):
    return conf > 0.5

# -----------------------------
# Setup plot
# -----------------------------
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(windows[0])
ax.set_ylim(np.min(windows)-0.1, np.max(windows)+0.1)
ax.set_title("Live ECG Dashboard (Local Record)")

# -----------------------------
# Realistic streaming loop
# -----------------------------
for i, window in enumerate(realtime_stream(signal, fs, delay=0.01)):
    # Slow it down to human-like speed
    time.sleep(0.1)  # controls visual speed

    # Small optional noise
    noisy_window = window + np.random.normal(0, 0.002, size=window.shape)
    line.set_ydata(noisy_window)

    conf = dummy_prediction(i)
    if check_alert(conf):
        ax.set_facecolor("mistyrose")
        print(f"ALERT: Abnormal in window {i+1} | Conf: {conf:.2f}")
    else:
        ax.set_facecolor("white")

    ax.set_title(f"Window {i+1}/{num_windows} | Conf: {conf:.2f}")
    fig.canvas.draw()
    fig.canvas.flush_events()

    if i >= 50:  # limit for testing
        break

plt.ioff()
plt.show()

# -----------------------------
# Optional Comparison Mode
# -----------------------------
if show_comparison:
    plt.figure(figsize=(10,4))
    plt.subplot(2,1,1)
    plt.plot(windows[10])
    plt.title("Abnormal Window (Dummy)")
    plt.subplot(2,1,2)
    plt.plot(windows[0])
    plt.title("Normal Reference Window")
    plt.tight_layout()
    plt.show()
'''
import numpy as np
import matplotlib.pyplot as plt
import time

# -----------------------------
# Config
# -----------------------------
all_windows = np.load("all_windows.npy", allow_pickle=True)
all_labels = np.load("all_labels.npy", allow_pickle=True)
delay = 0.5  # seconds per window

# Dummy alert function
def dummy_alert(label):
    # 1 = abnormal, 0 = normal
    return label == 1

# -----------------------------
# Live streaming simulation
# -----------------------------
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(all_windows[0].squeeze())
ax.set_ylim(np.min(all_windows), np.max(all_windows))
ax.set_title("Live ECG Streaming (Dummy Alerts)")

for i, window in enumerate(all_windows[:100]):  # first 100 windows for demo
    line.set_ydata(window.squeeze())

    # Dummy alert
    if dummy_alert(all_labels[i]):
        ax.set_facecolor("mistyrose")
        print(f"ALERT: Abnormal window {i+1}")
    else:
        ax.set_facecolor("white")

    ax.set_title(f"Window {i+1}/{len(all_windows)}")
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(delay)

plt.ioff()
plt.show()
