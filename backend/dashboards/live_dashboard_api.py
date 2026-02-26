from flask import Flask, jsonify, request
import numpy as np
import random
from flask_cors import CORS

CLASS_NAMES = {
    0: "N",
    1: "S",
    2: "V",
    3: "F",
    4: "Q",
}

# -----------------------------
# Load saved dataset + metadata
# -----------------------------
all_windows = np.load("merged_all_windows.npy",  mmap_mode="r",allow_pickle=True)
all_labels = np.load("merged_all_labels.npy", allow_pickle=True)
all_record_names = np.load("merged_all_record_names.npy", allow_pickle=True)
all_lead_names = np.load("merged_all_lead_names.npy", allow_pickle=True)

app = Flask(__name__)
CORS(app)


def build_mask(record_name=None, lead_name=None):
    mask = np.ones(len(all_windows), dtype=bool)
    if record_name is not None:
        mask &= (all_record_names == record_name)
    if lead_name is not None:
        mask &= (all_lead_names == lead_name)
    return mask


# -----------------------------
# API: fetch one window
# Query params supported:
#   ?record_name=100&lead_name=MLII
# -----------------------------
@app.route("/get_window/<int:idx>")
def get_window(idx):
    record_name = request.args.get("record_name")
    lead_name = request.args.get("lead_name")
    mask = build_mask(record_name=record_name, lead_name=lead_name)
    filtered_idx = np.where(mask)[0]

    if idx < 0 or idx >= len(filtered_idx):
        return jsonify({"error": "Index out of range for selected filter"}), 400

    real_idx = int(filtered_idx[idx])
    window = all_windows[real_idx].squeeze().tolist()
    raw_label = all_labels[real_idx]
    if raw_label is None:
        label = None
    else:
        try:
            label = int(raw_label)
        except (TypeError, ValueError):
            return jsonify({
                "error": "Invalid label value in dataset",
                "global_idx": real_idx,
                "raw_label": str(raw_label)
            }), 500

    return jsonify({
        "window_idx": idx,
        "global_idx": real_idx,
        "label": label,
        "class_name": CLASS_NAMES.get(label, None) if label is not None else None,
        "record_name": str(all_record_names[real_idx]),
        "lead_name": str(all_lead_names[real_idx]),
        "data": window
    })

@app.route("/predict", methods=["POST"])
def predict():
    fake_classes = ["N", "S", "V", "F", "Q"]

    # fake_classes = ["Normal", "MI", "STTC", "CD", "HYP"]
    
    prediction = random.choice(fake_classes)
    confidence = round(random.uniform(0.75, 0.99), 2)

    return jsonify({
        "class": prediction,
        "confidence": confidence
    })
# -----------------------------
# API: metadata summary
# -----------------------------
@app.route("/metadata")
def metadata():
    return jsonify({
        "total_windows": int(len(all_windows)),
        "records": sorted(np.unique(all_record_names).astype(str).tolist()),
        "leads": sorted(np.unique(all_lead_names).astype(str).tolist()),
        "class_map": CLASS_NAMES
    })


if __name__ == "__main__":
    app.run(debug=True)