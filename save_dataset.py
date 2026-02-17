import time
import numpy as np
import matplotlib.pyplot as plt

from src.data_loader import get_record_list, load_record_all_leads
from src.pipeline import process_signal
from src.annotation_loader import load_annotations
from src.window_labeler import label_windows

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
data_path = "data/mitdb"
window_size = 720  # 2-second window
stride = 360       # 50% overlap

# -----------------------------
# 1) Get all records
# -----------------------------
records = get_record_list(data_path)
print(f"Total records found: {len(records)}")

# -----------------------------
# 2) Process all records and all leads
# -----------------------------
all_windows = []
all_labels = []
all_record_names = []
all_lead_names = []

for idx, record_name in enumerate(records):
    print(f"\nProcessing record {record_name} ({idx + 1}/{len(records)})")

    signals, fs, lead_names = load_record_all_leads(data_path, record_name)
    samples, symbols = load_annotations(data_path, record_name)

    # Labels are based on time indices, so they can be reused for every lead in this record.
    labels = label_windows(
        samples,
        symbols,
        signal_length=signals.shape[0],
        window_size=window_size,
        stride=stride
    )
    labels = np.array(labels, dtype=object)
    valid_idx = labels != None

    for lead_idx, lead_name in enumerate(lead_names):
        signal = signals[:, lead_idx]

        start_time = time.time()
        windows = process_signal(signal, fs, window_size=window_size, stride=stride)
        end_time = time.time()

        if len(windows) != len(labels):
            raise ValueError(
                f"Window/label length mismatch for record={record_name}, lead={lead_name}: "
                f"{len(windows)} windows vs {len(labels)} labels"
            )

        filtered_windows = windows[valid_idx]
        filtered_labels = labels[valid_idx]

        all_windows.append(filtered_windows)
        all_labels.append(filtered_labels)
        all_record_names.append(np.full(len(filtered_labels), record_name, dtype=object))
        all_lead_names.append(np.full(len(filtered_labels), lead_name, dtype=object))

        print(
            f"Lead {lead_name}: windows={len(filtered_windows)} "
            f"| N={np.sum(filtered_labels == 0)} "
            f"| S={np.sum(filtered_labels == 1)} "
            f"| V={np.sum(filtered_labels == 2)} "
            f"| F={np.sum(filtered_labels == 3)} "
            f"| Q={np.sum(filtered_labels == 4)} "
            f"| Time={round(end_time - start_time, 4)}s"
        )

# -----------------------------
# 3) Combine all records/leads
# -----------------------------
if len(all_windows) == 0:
    raise RuntimeError(
        "No valid windows were generated. Check window_size/stride, label mapping, and input records."
    )

all_windows = np.concatenate(all_windows, axis=0)
all_labels = np.concatenate(all_labels, axis=0)
all_record_names = np.concatenate(all_record_names, axis=0)
all_lead_names = np.concatenate(all_lead_names, axis=0)

if all_windows.shape[0] == 0:
    raise RuntimeError(
        "All generated arrays are empty after filtering labels. "
        "Relax mapping/filtering or review annotations."
    )

print("\nAll records and leads processed")
print(f"Total windows: {all_windows.shape[0]}")
print(f"Unique leads: {len(np.unique(all_lead_names))}")
print(
    f"Class counts -> "
    f"N={np.sum(all_labels == 0)}, "
    f"S={np.sum(all_labels == 1)}, "
    f"V={np.sum(all_labels == 2)}, "
    f"F={np.sum(all_labels == 3)}, "
    f"Q={np.sum(all_labels == 4)}"
)

# -----------------------------
# 4) Save dataset + metadata
# -----------------------------
np.save("all_windows.npy", all_windows)
np.save("all_labels.npy", all_labels)
np.save("all_record_names.npy", all_record_names)
np.save("all_lead_names.npy", all_lead_names)
print("Saved: all_windows.npy, all_labels.npy, all_record_names.npy, all_lead_names.npy")

# -----------------------------
# 5) Sanity check plots
# -----------------------------
normal_candidates = np.where(all_labels == 0)[0]
abnormal_candidates = np.where(all_labels != 0)[0]

if len(normal_candidates) > 0 and len(abnormal_candidates) > 0:
    normal_idx = normal_candidates[0]
    abnormal_idx = abnormal_candidates[0]

    plt.figure(figsize=(10, 4))
    plt.subplot(2, 1, 1)
    plt.plot(all_windows[normal_idx])
    normal_class = int(all_labels[normal_idx])
    plt.title(
        f"Sample {CLASS_NAMES.get(normal_class, str(normal_class))} Window "
        f"({all_record_names[normal_idx]} - {all_lead_names[normal_idx]})"
    )

    plt.subplot(2, 1, 2)
    plt.plot(all_windows[abnormal_idx])
    abnormal_class = int(all_labels[abnormal_idx])
    plt.title(
        f"Sample {CLASS_NAMES.get(abnormal_class, str(abnormal_class))} Window "
        f"({all_record_names[abnormal_idx]} - {all_lead_names[abnormal_idx]})"
    )
    plt.tight_layout()
    plt.show()
else:
    print("Skipped plot: normal or abnormal class not found in dataset.")
