import re
import tempfile
import time
from pathlib import Path

import wfdb
from flask import Flask, jsonify, request
import numpy as np
import random
from flask_cors import CORS
from src.preprocessing import bandpass_filter, normalize_signal

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

LEAD_NAME_ALIASES = {
    "MLII": "II",
    "AVR": "aVR",
    "AVL": "aVL",
    "AVF": "aVF",
}

STANDARD_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
MAX_PREVIEW_POINTS = 3000


def normalize_lead_name(raw_name, fallback_index):
    lead = str(raw_name).strip()
    if not lead:
        return f"Lead_{fallback_index + 1}"

    compact = re.sub(r"\s+", "", lead)
    upper = compact.upper()
    aliased = LEAD_NAME_ALIASES.get(upper, upper)

    if aliased in {"I", "II", "III"} or re.fullmatch(r"V\d+", aliased):
        return aliased

    return aliased if aliased.startswith("aV") else compact


def lead_sort_key(lead_name):
    if lead_name in STANDARD_LEADS:
        return (0, STANDARD_LEADS.index(lead_name), lead_name)
    return (1, 999, lead_name)


def find_local_record_prefix(stem_name):
    search_roots = [
        Path("data/mitdb"),
        Path("data/incartdb"),
        Path("../data/mitdb"),
        Path("../data/incartdb"),
    ]
    for root in search_roots:
        prefix = root / stem_name
        if prefix.with_suffix(".dat").exists() and prefix.with_suffix(".hea").exists():
            return prefix
    return None


def load_wfdb_record_from_upload():
    ecg_file = request.files.get("ecg_file")
    if ecg_file is None or not ecg_file.filename:
        raise ValueError("Missing uploaded ECG file")

    suffix = Path(ecg_file.filename).suffix.lower()
    if suffix != ".dat":
        raise ValueError("Only .dat files are supported by this endpoint")

    stem_name = Path(ecg_file.filename).stem
    header_file = request.files.get("header_file")

    if header_file is not None and header_file.filename:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            dat_path = tmp_dir_path / f"{stem_name}.dat"
            hea_path = tmp_dir_path / f"{stem_name}.hea"
            ecg_file.save(dat_path)
            header_file.save(hea_path)
            return wfdb.rdrecord(str(tmp_dir_path / stem_name))

    local_prefix = find_local_record_prefix(stem_name)
    if local_prefix is None:
        raise ValueError(
            "Missing .hea for this .dat file. Upload matching .hea as header_file or use a record available in backend/data."
        )
    return wfdb.rdrecord(str(local_prefix))


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
    start_ts = time.perf_counter()
    payload = request.get_json(silent=True) or {}
    raw_signal = payload.get("signal", [])

    signal = np.asarray(raw_signal, dtype=float) if isinstance(raw_signal, list) else np.asarray([], dtype=float)
    if signal.size > 0:
        finite_mask = np.isfinite(signal)
        signal = signal[finite_mask]

    fake_classes = ["N", "S", "V", "F", "Q"]

    # fake_classes = ["Normal", "MI", "STTC", "CD", "HYP"]
    
    prediction = random.choice(fake_classes)
    confidence = round(random.uniform(0.75, 0.99), 2)
    inference_time_ms = (time.perf_counter() - start_ts) * 1000.0

    heatmap = []
    if signal.size > 0:
        centered = signal - float(np.mean(signal))
        denom = float(np.max(np.abs(centered))) if np.max(np.abs(centered)) > 0 else 1.0
        heatmap = (centered / denom).astype(float).tolist()

    status = "Normal" if prediction == "N" else "Abnormal"

    return jsonify({
        "class": prediction,
        "confidence": confidence,
        "status": status,
        "inference_time_ms": round(inference_time_ms, 3),
        "heatmap": heatmap,
    })


@app.route("/preprocess_signal", methods=["POST"])
def preprocess_signal():
    payload = request.get_json(silent=True) or {}
    signal_raw = payload.get("signal", [])
    fs_raw = payload.get("fs", 360.0)

    if not isinstance(signal_raw, list) or len(signal_raw) < 20:
        return jsonify({"error": "signal must be a numeric list with enough samples"}), 400

    try:
        fs = float(fs_raw)
        signal = np.asarray(signal_raw, dtype=float)
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid signal or fs"}), 400

    finite_mask = np.isfinite(signal)
    if not np.any(finite_mask):
        return jsonify({"error": "Signal has no finite values"}), 400
    signal = signal[finite_mask]

    try:
        filtered = bandpass_filter(signal, fs, lowcut=0.5, highcut=40, order=4)
        normalized = normalize_signal(filtered)
    except Exception as ex:
        return jsonify({"error": f"Preprocessing failed: {str(ex)}"}), 400

    return jsonify({
        "processed_signal": normalized.astype(float).tolist(),
        "fs": fs,
        "pipeline": "bandpass(0.5-40Hz)+zscore",
    })


@app.route("/preview_uploaded_signal", methods=["POST"])
def preview_uploaded_signal():
    try:
        record = load_wfdb_record_from_upload()
    except ValueError as ex:
        return jsonify({"error": str(ex)}), 400
    except Exception as ex:
        return jsonify({"error": f"Unable to read WFDB record: {str(ex)}"}), 500

    signals = record.p_signal
    if signals is None:
        return jsonify({"error": "Record has no p_signal data"}), 400
    if signals.ndim == 1:
        signals = signals[:, np.newaxis]

    sample_count = int(signals.shape[0])
    fs = float(record.fs) if record.fs else 360.0

    if sample_count <= MAX_PREVIEW_POINTS:
        indices = np.arange(sample_count, dtype=int)
    else:
        indices = np.linspace(0, sample_count - 1, MAX_PREVIEW_POINTS, dtype=int)
        indices = np.unique(indices)

    time_axis = (indices / fs).astype(float).tolist()
    raw_lead_names = record.sig_name if record.sig_name is not None else [f"lead_{i + 1}" for i in range(signals.shape[1])]

    lead_series = {}
    for lead_idx, raw_name in enumerate(raw_lead_names):
        lead_name = normalize_lead_name(raw_name, lead_idx)
        values = signals[indices, lead_idx].astype(float).tolist()
        if lead_name in lead_series:
            lead_name = f"{lead_name}_{lead_idx + 1}"
        lead_series[lead_name] = values

    available_leads = sorted(lead_series.keys(), key=lead_sort_key)
    default_lead = "II" if "II" in lead_series else available_leads[0]

    return jsonify({
        "fs": fs,
        "total_samples": sample_count,
        "time": time_axis,
        "available_leads": available_leads,
        "default_lead": default_lead,
        "lead_series": lead_series,
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
