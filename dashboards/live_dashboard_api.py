'''
import time
import matplotlib.pyplot as plt
import requests
import numpy as np

# -----------------------------
# Config
# -----------------------------
API_URL = "http://127.0.0.1:5000/get_window/{}"  # Flask API endpoint
NUM_WINDOWS = 50                               # How many windows to fetch
DELAY = 0.1                                    # seconds between windows

# -----------------------------
# Dummy prediction helper
# -----------------------------
def dummy_prediction(label):
    """
    Simulate prediction confidence.
    label=0 -> normal
    label=1 -> abnormal
    """
    if label == 1:
        return np.random.uniform(0.7, 1.0)  # high confidence abnormal
    else:
        return np.random.uniform(0.0, 0.3)  # low confidence normal

# -----------------------------
# Fetch a window from API
# -----------------------------
def get_window(idx):
    try:
        response = requests.get(API_URL.format(idx))
        data = response.json()
        return np.array(data["data"]), data["label"]
    except Exception as e:
        print(f"Error fetching window {idx}: {e}")
        return None, None

# -----------------------------
# Setup plot
# -----------------------------
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(np.zeros(432))  # initial dummy line
ax.set_ylim(-2, 8)              # adjust based on your data
ax.set_title("Live ECG Streaming via Flask API")

# -----------------------------
# Real-time simulation loop
# -----------------------------
for i in range(NUM_WINDOWS):
    window, label = get_window(i)
    if window is None:
        continue

    # Add some noise for robustness testing
    noisy_window = window + np.random.normal(0, 0.02, size=window.shape)

    # Update line
    line.set_ydata(noisy_window)

    # Dummy prediction confidence
    conf = dummy_prediction(label)

    # Color the background based on alert
    if conf > 0.5:
        ax.set_facecolor("mistyrose")  # abnormal
        print(f"ALERT: Abnormal detected in window {i} (confidence {conf:.2f})")
    else:
        ax.set_facecolor("white")      # normal

    ax.set_title(f"Window {i+1}/{NUM_WINDOWS} | Confidence: {conf:.2f}")
    fig.canvas.draw()
    fig.canvas.flush_events()

    time.sleep(DELAY)

plt.ioff()
plt.show()
'''
'''
import time
import matplotlib.pyplot as plt
import numpy as np
import requests

# -----------------------------
# Config
# -----------------------------
API_URL = "http://127.0.0.1:5000/get_window/{}"  # Flask API endpoint
NUM_WINDOWS = 50
DELAY = 0.1
show_comparison = False

# -----------------------------
# Helper functions
# -----------------------------
def get_window(idx):
    try:
        response = requests.get(API_URL.format(idx))
        data = response.json()
        return np.array(data["data"]), data["label"]
    except Exception as e:
        print(f"Error fetching window {idx}: {e}")
        return None, None

def dummy_prediction(label):
    if label == 1:
        return np.random.uniform(0.7, 1.0)
    else:
        return np.random.uniform(0.0, 0.3)

def check_alert(conf):
    return conf > 0.5

# -----------------------------
# Initialize plot
# -----------------------------
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(np.zeros(720))
#ax.set_ylim(-1, 1)
ax.set_ylim(-2, 8) 
ax.set_title("Live ECG Dashboard (API-driven)")

# -----------------------------
# Real-time streaming loop
# -----------------------------
for i in range(NUM_WINDOWS):
    window, label = get_window(i)
    if window is None:
        continue

    # Optional tiny noise
    window = window + np.random.normal(0, 0.005, size=window.shape)

    line.set_ydata(window)
    conf = dummy_prediction(label)

    if check_alert(conf):
        ax.set_facecolor("mistyrose")
        print(f"ALERT: Abnormal detected in window {i+1} | Confidence: {conf:.2f}")
    else:
        ax.set_facecolor("white")

    ax.set_title(f"Window {i+1}/{NUM_WINDOWS} | Confidence: {conf:.2f}")
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(DELAY)

plt.ioff()
plt.show()

# -----------------------------
# Optional: Comparison Mode
# -----------------------------
if show_comparison:
    plt.figure(figsize=(10,4))
    abnormal_idx = 10
    normal_idx = 0
    # Fetch windows from API
    w1, _ = get_window(abnormal_idx)
    w2, _ = get_window(normal_idx)

    plt.subplot(2,1,1)
    plt.plot(w1)
    plt.title("Abnormal Window (Dummy)")

    plt.subplot(2,1,2)
    plt.plot(w2)
    plt.title("Normal Reference Window")
    plt.tight_layout()
    plt.show()
'''

from flask import Flask, jsonify
import numpy as np

# -----------------------------
# Load saved dataset
# -----------------------------
all_windows = np.load("all_windows.npy", allow_pickle=True)
all_labels = np.load("all_labels.npy", allow_pickle=True)

app = Flask(__name__)

# -----------------------------
# API endpoint to fetch a window
# -----------------------------
@app.route("/get_window/<int:idx>")
def get_window(idx):
    if idx < 0 or idx >= len(all_windows):
        return jsonify({"error": "Index out of range"})
    
    window = all_windows[idx].squeeze().tolist()
    label = int(all_labels[idx])
    
    return jsonify({
        "window_idx": idx,
        "label": label,
        "data": window
    })

# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
