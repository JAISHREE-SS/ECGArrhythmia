# ECG Project

This repo now runs from the repository root with application code and data under `backend/`.

---

## Runtime Contract

- Working directory: `Z:\ECG_PROJECT`
- Python interpreter: `backend\ecg_env\Scripts\python.exe`
- Scripts: `backend\*.py`
- Data roots: `backend\data\mitdb`, `backend\data\incartdb`
- Outputs: `backend\all_*.npy`, `backend\class_map.json`, `backend\dataset_meta.json`

---

## Verify Environment

```powershell
Get-Location
Test-Path backend\ecg_env\Scripts\python.exe
Test-Path backend\data\mitdb
Test-Path backend\data\incartdb
```

---

## Build Datasets (Root-Based)

### 1) MIT dataset (prefixed)

```powershell
.\backend\ecg_env\Scripts\python.exe backend\save_dataset.py --output-prefix backend\mit_ --no-plot
```

Note: lead alias normalization is applied while saving (`MLII` is saved as `II`).

### 2) INCART dataset at 360 Hz (prefixed)

```powershell
.\backend\ecg_env\Scripts\python.exe backend\save_dataset_incart.py --data-path backend\data\incartdb --output-prefix backend\incart_ --target-fs 360 --no-plot
```

Note: lead alias normalization is also applied for INCART (`MLII` is saved as `II`).

### 3) Merge MIT + INCART (strict dedup by record name)

```powershell
.\backend\ecg_env\Scripts\python.exe backend\merge_datasets.py --mit-prefix backend\mit_ --incart-prefix backend\incart_ --output-prefix backend\ --on-duplicate-record error
```

If you intentionally want to skip duplicates from INCART:

```powershell
.\backend\ecg_env\Scripts\python.exe backend\merge_datasets.py --mit-prefix backend\mit_ --incart-prefix backend\incart_ --output-prefix backend\ --on-duplicate-record skip-second
```

### 4) Merge into explicit `merged_all_*` files

```powershell
.\backend\ecg_env\Scripts\python.exe backend\merge_datasets.py --mit-prefix backend\mit_ --incart-prefix backend\incart_ --output-prefix backend\merged_
```

---

## Run API and Dashboard

`live_dashboard_api.py` and `test_ecg_dashboard.py` load `all_*.npy` from current directory.  
Run them from inside `backend/`.

### API

```powershell
cd backend
..\backend\ecg_env\Scripts\python.exe -m dashboards.live_dashboard_api
```

Default URL: `http://127.0.0.1:5000`

Smoke checks:
- `GET http://127.0.0.1:5000/metadata`
- `GET http://127.0.0.1:5000/get_window/0`

### Local dashboard simulator

```powershell
cd backend
..\backend\ecg_env\Scripts\python.exe dashboards\test_ecg_dashboard.py
```

---

## Validate Outputs

Confirm merged files exist (default merge output):
- `backend\all_windows.npy`
- `backend\all_labels.npy`
- `backend\all_record_names.npy`
- `backend\all_lead_names.npy`

If you used `--output-prefix backend\merged_`, confirm:
- `backend\merged_all_windows.npy`
- `backend\merged_all_labels.npy`
- `backend\merged_all_record_names.npy`
- `backend\merged_all_lead_names.npy`

Check metadata:

```powershell
.\backend\ecg_env\Scripts\python.exe -c "import json; print(json.load(open('backend/dataset_meta.json')))"
```

Check shape:

```powershell
.\backend\ecg_env\Scripts\python.exe -c "import numpy as np; w=np.load('backend/all_windows.npy', mmap_mode='r'); print(w.shape, w[0].shape)"
```

Expected for INCART default window config at 360 Hz:
- single window shape `(720, 1)`

---

## Troubleshooting

- `ModuleNotFoundError: dashboards`
  - Start API from `backend/`.
- `FileNotFoundError: all_windows.npy`
  - Ensure merge output uses `--output-prefix backend\` and run API from `backend/`.
- `wfdb` import errors
  - Use `backend\ecg_env\Scripts\python.exe`, not system Python.
- Duplicate record conflict on merge
  - Keep `--on-duplicate-record error` unless you explicitly want `skip-second`.

---

## Git Push Strategy

This is a single Git repository, so `git push origin main` pushes committed history for the whole repo.

### Push backend changes to current GitHub repo

```powershell
git add backend README.md
git commit -m "Backend updates"
git push origin main
```

### Push `frontend/` to a separate Lovable-connected repo

```powershell
git remote add lovable <FRONTEND_REPO_URL>
git subtree split --prefix frontend -b frontend-sync
git push lovable frontend-sync:main
git branch -D frontend-sync
```

Notes:
- `git remote add lovable ...` is needed once per local clone.
- If the Lovable repo rejects non-fast-forward pushes, use `git push lovable frontend-sync:main --force` only if intentional.
