# ECG Project

ECG preprocessing and streaming utilities for arrhythmia datasets.

This repository supports:
- MIT-BIH dataset preprocessing
- INCART dataset preprocessing with optional resampling to 360 Hz
- Merging MIT and INCART outputs into one dataset without duplicate records
- Dashboard/API playback from saved `.npy` windows

---

## Project Structure

```text
ECG_PROJECT/
|-- data/
|   |-- mitdb/                  # MIT-BIH files (.dat, .hea, .atr)
|   |-- incartdb/               # INCART files (.dat, .hea, .atr)
|-- src/
|-- dashboards/
|-- save_dataset.py             # Build MIT dataset
|-- download_incart_data.py     # Download INCART from PhysioNet
|-- save_dataset_incart.py      # Build INCART dataset (supports --target-fs)
|-- merge_datasets.py           # Merge MIT + INCART with dedup by record
|-- requirements.txt
|-- README.md
```

---

## Label Scheme

`src/label_mapper.py` maps beat symbols into:
- `0 -> N`
- `1 -> S`
- `2 -> V`
- `3 -> F`
- `4 -> Q`

---

## Setup

```powershell
python -m venv ecg_env
.\ecg_env\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Usage

### 1) Build MIT dataset (recommended with prefix)

```powershell
python save_dataset.py --output-prefix mit_ --no-plot
```

Outputs:
- `mit_all_windows.npy`
- `mit_all_labels.npy`
- `mit_all_record_names.npy`
- `mit_all_lead_names.npy`

### 2) Download INCART

```powershell
python download_incart_data.py --target-dir data/incartdb
```

### 3) Build INCART dataset at 360 Hz

```powershell
python save_dataset_incart.py --output-prefix incart_ --target-fs 360 --no-plot
```

Outputs:
- `incart_all_windows.npy`
- `incart_all_labels.npy`
- `incart_all_record_names.npy`
- `incart_all_lead_names.npy`
- `incart_class_map.json`
- `incart_dataset_meta.json`

Default window settings are 2.0 s window and 1.0 s stride, so at 360 Hz:
- `window_size_samples = 720`
- `stride_samples = 360`

### 4) Merge MIT + INCART without duplicate records

Strict mode (fail if duplicate record names exist):
```powershell
python merge_datasets.py --mit-prefix mit_ --incart-prefix incart_ --on-duplicate-record error
```

Skip duplicates from second dataset (INCART):
```powershell
python merge_datasets.py --mit-prefix mit_ --incart-prefix incart_ --on-duplicate-record skip-second
```

By default, merged output is written to:
- `all_windows.npy`
- `all_labels.npy`
- `all_record_names.npy`
- `all_lead_names.npy`
- `class_map.json` (if available)
- `dataset_meta.json`

### 5) Run API

```powershell
python -m dashboards.live_dashboard_api
```

Default URL: `http://127.0.0.1:5000`

Endpoints:
- `GET /metadata`
- `GET /get_window/<idx>`

Optional query params:
- `record_name`
- `lead_name`

Example:
```text
GET /get_window/0?record_name=100&lead_name=MLII
```

---

## Notes

- Use prefixes (`mit_`, `incart_`) to avoid overwriting datasets.
- INCART source files (`.dat/.hea/.atr`) are not modified; only output `.npy/.json` files are generated.
- `dataset_meta.json` records effective sampling/window settings for traceability.
