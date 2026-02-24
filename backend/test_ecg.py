import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1) Load saved dataset + metadata
# -----------------------------
all_windows = np.load("all_windows.npy", allow_pickle=True)
all_labels = np.load("all_labels.npy", allow_pickle=True)
all_record_names = np.load("all_record_names.npy", allow_pickle=True)
all_lead_names = np.load("all_lead_names.npy", allow_pickle=True)

print("Dataset loaded")
print("Total windows:", all_windows.shape[0])
print("Normal:", np.sum(all_labels == 0))
print("Abnormal:", np.sum(all_labels == 1))
print("Unique records:", len(np.unique(all_record_names)))
print("Unique leads:", len(np.unique(all_lead_names)))

# -----------------------------
# 2) Quick lead-wise stats
# -----------------------------
unique_leads = np.unique(all_lead_names)
print("\nPer-lead window counts:")
for lead in unique_leads:
    lead_idx = all_lead_names == lead
    lead_labels = all_labels[lead_idx]
    print(
        f"{lead}: total={np.sum(lead_idx)} "
        f"| normal={np.sum(lead_labels == 0)} "
        f"| abnormal={np.sum(lead_labels == 1)}"
    )

# -----------------------------
# 3) Sanity check plots with lead/record info
# -----------------------------
normal_candidates = np.where(all_labels == 0)[0]
abnormal_candidates = np.where(all_labels == 1)[0]

if len(normal_candidates) > 0 and len(abnormal_candidates) > 0:
    normal_idx = normal_candidates[0]
    abnormal_idx = abnormal_candidates[0]

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(all_windows[normal_idx].squeeze())
    plt.title(
        f"Normal | Record={all_record_names[normal_idx]} | Lead={all_lead_names[normal_idx]}"
    )

    plt.subplot(2, 1, 2)
    plt.plot(all_windows[abnormal_idx].squeeze())
    plt.title(
        f"Abnormal | Record={all_record_names[abnormal_idx]} | Lead={all_lead_names[abnormal_idx]}"
    )

    plt.tight_layout()
    plt.show()
else:
    print("Skipped plotting: normal or abnormal class is missing.")

# -----------------------------
# 4) Inspect first few windows with metadata
# -----------------------------
print("\nFirst 5 entries:")
for i in range(min(5, len(all_labels))):
    print(
        f"idx={i} | label={all_labels[i]} "
        f"| record={all_record_names[i]} | lead={all_lead_names[i]}"
    )

print("Window shape:", all_windows[0].shape)
