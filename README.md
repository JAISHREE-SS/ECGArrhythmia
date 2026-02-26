# ECG Arrhythmia Detection System

This repository contains a full-stack ECG analysis project:
- `backend/`: Python data pipeline and Flask APIs for ECG windowing, inference, and live dashboard endpoints.
- `frontend/`: React + TypeScript dashboard for upload, monitoring, history, and model insights.

## Repository Structure

- `backend/dashboards/live_dashboard_api.py`: live dashboard API server.
- `backend/dashboards/flask_api.py`: additional Flask API entrypoint.
- `backend/save_dataset.py`: MIT dataset preprocessing.
- `backend/save_dataset_incart.py`: INCART dataset preprocessing.
- `backend/merge_datasets.py`: merge generated dataset artifacts.
- `frontend/src/pages/`: dashboard pages (`Dashboard`, `UploadECG`, `RealTimeMonitor`, `PredictionHistory`, `ModelInsights`).

## Backend Setup (Windows)

1. Use the project virtual environment:

```powershell
backend\ecg_env\Scripts\python.exe --version
```

2. Install backend dependencies (if needed):

```powershell
backend\ecg_env\Scripts\python.exe -m pip install -r backend\requirements.txt
```

## Frontend Setup

1. Install dependencies:

```powershell
cd frontend
npm install
```

2. Start dev server:

```powershell
npm run dev
```

## Run Backend API

From repository root:

```powershell
cd backend
..\backend\ecg_env\Scripts\python.exe -m dashboards.live_dashboard_api
```

Default local URL: `http://127.0.0.1:5000`

Useful endpoints:
- `GET /metadata`
- `GET /get_window/<index>`

## Dataset Build Workflow

From repository root:

```powershell
backend\ecg_env\Scripts\python.exe backend\save_dataset.py --output-prefix backend\mit_ --no-plot
backend\ecg_env\Scripts\python.exe backend\save_dataset_incart.py --data-path backend\data\incartdb --output-prefix backend\incart_ --target-fs 360 --no-plot
backend\ecg_env\Scripts\python.exe backend\merge_datasets.py --mit-prefix backend\mit_ --incart-prefix backend\incart_ --output-prefix backend\merged_
```

Expected merged outputs:
- `backend/merged_all_windows.npy`
- `backend/merged_all_labels.npy`
- `backend/merged_all_record_names.npy`
- `backend/merged_all_lead_names.npy`

## Notes

- Run dashboard-related Python modules from inside `backend/` so local imports resolve correctly.
- `frontend/node_modules` and build outputs should stay uncommitted.
