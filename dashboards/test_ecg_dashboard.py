import time
import numpy as np
import matplotlib.pyplot as plt

CLASS_NAMES = {
    0: "N",
    1: "S",
    2: "V",
    3: "F",
    4: "Q",
}

# -----------------------------
# Config
# -----------------------------
delay = 0.5  # seconds per window
max_windows = 100

# Optional filters (set to None to disable)
record_filter = None  # e.g. "100"
lead_filter = None    # e.g. "MLII"

# -----------------------------
# Load saved dataset + metadata
# -----------------------------
all_windows = np.load("all_windows.npy", allow_pickle=True)
all_labels = np.load("all_labels.npy", allow_pickle=True)
all_record_names = np.load("all_record_names.npy", allow_pickle=True)
all_lead_names = np.load("all_lead_names.npy", allow_pickle=True)

mask = np.ones(len(all_windows), dtype=bool)
if record_filter is not None:
    mask &= (all_record_names == record_filter)
if lead_filter is not None:
    mask &= (all_lead_names == lead_filter)

windows = all_windows[mask]
labels = all_labels[mask]
record_names = all_record_names[mask]
lead_names = all_lead_names[mask]

if len(windows) == 0:
    raise ValueError("No windows found for selected filters.")

print(f"Loaded windows: {len(windows)}")
print(f"Unique records: {len(np.unique(record_names))}")
print(f"Unique leads: {len(np.unique(lead_names))}")


def dummy_alert(label):
    return label != 0


# -----------------------------
# Live streaming simulation
# -----------------------------
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(windows[0].squeeze())
ax.set_ylim(np.min(windows), np.max(windows))
ax.set_title("Live ECG Streaming (Lead-aware)")

num_to_show = min(max_windows, len(windows))
for i in range(num_to_show):
    window = windows[i]
    label = labels[i]
    record_name = record_names[i]
    lead_name = lead_names[i]

    line.set_ydata(window.squeeze())

    if dummy_alert(label):
        ax.set_facecolor("mistyrose")
        status = "ABNORMAL"
    else:
        ax.set_facecolor("white")
        status = "NORMAL"

    try:
        label_int = int(label)
    except (TypeError, ValueError):
        label_int = None
    class_name = CLASS_NAMES.get(label_int, f"Class-{label}")

    ax.set_title(
        f"Window {i + 1}/{num_to_show} | {status} ({class_name}) | "
        f"Record={record_name} | Lead={lead_name}"
    )
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(delay)

plt.ioff()
plt.show()
