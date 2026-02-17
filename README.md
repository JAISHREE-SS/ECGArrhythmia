
# ECG Project

ECG data preprocessing and streaming utilities for MIT-BIH arrhythmia experiments.

This repository provides:
- A preprocessing pipeline from raw MIT-BIH records to windowed `.npy` datasets
- A local dashboard script to simulate live ECG window playback
- A Flask API to stream windows with optional record/lead filters
- Support for training models, testing dashboards, or integrating live predictions

---

## Project Structure

```text
ECG_PROJECT/
|-- data/mitdb/                  # MIT-BIH files (.dat, .hea, .atr)
|-- src/
|   |-- annotation_loader.py
|   |-- data_loader.py
|   |-- label_mapper.py
|   |-- pipeline.py
|   |-- preprocessing.py
|   |-- realtime_sim.py
|   |-- sliding_window.py
|   |-- window_labeler.py
|-- dashboards/
|   |-- live_dashboard_api.py    # Main Flask API
|   |-- test_ecg_dashboard.py    # Matplotlib live playback simulator
|   |-- flask_api.py             # Legacy dummy API (not primary)
|-- save_dataset.py              # Build dataset from raw MIT-BIH files
|-- test_ecg.py                  # Dataset sanity checks
|-- all_windows.npy              # Generated windows
|-- all_labels.npy               # Class labels (0..4)
|-- all_record_names.npy         # Record name per window
|-- all_lead_names.npy           # Lead name per window
|-- requirements.txt
|-- README.md
````

---

## Label Scheme

`src/label_mapper.py` maps beat symbols into 5 main classes:

* `0 -> N` (normal-like)
* `1 -> S` (supraventricular ectopic)
* `2 -> V` (ventricular ectopic)
* `3 -> F` (fusion)
* `4 -> Q` (paced/unclassifiable)

---

## Setup

```powershell
python -m venv ecg_env
.\ecg_env\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Usage

### 1) Generate dataset from raw MIT-BIH data

Requires MIT-BIH files under `data/mitdb/`.

```powershell
python save_dataset.py
```

Outputs:

* `all_windows.npy`
* `all_labels.npy`
* `all_record_names.npy`
* `all_lead_names.npy`

---

### 2) Run dataset sanity checks

```powershell
python test_ecg.py
```

---

### 3) Run live dashboard simulation

```powershell
python dashboards/test_ecg_dashboard.py
```

---

### 4) Run Flask API

```powershell
python -m dashboards.live_dashboard_api
```

Default URL: `http://127.0.0.1:5000`

Endpoints:

* `GET /metadata`
* `GET /get_window/<idx>`

Optional query params:

* `record_name`
* `lead_name`

Example:

```
GET /get_window/0?record_name=100&lead_name=MLII
```

Example response:

```json
{
  "window_idx": 0,
  "global_idx": 123,
  "label": 2,
  "class_name": "V",
  "record_name": "100",
  "lead_name": "MLII",
  "data": [0.01, -0.12, 0.08]
}
```

---

## Notes

* Use `dashboards/live_dashboard_api.py` as the primary API.
* Dashboard and API both require generated `.npy` dataset files.
* If class distribution looks incorrect, regenerate using `save_dataset.py`.

````

---

# 🚀 Now Finish The Rebase

After saving README:

```bash
git add README.md
git rebase --continue
````

Then:

```bash
git push origin main
```

Done.


