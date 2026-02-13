
# ECG Project 

This project contains **preprocessed MIT-BIH ECG dataset**, dashboards, and a Flask API for **arrhythmia detection simulation**.  

use this for **training models, testing dashboards, or integrating live predictions**.

---

## **1️⃣ Project Structure**

```

ECG_PROJECT/
│
├── data/mitdb/                  # MIT-BIH dataset files (.dat, .hea, .atr)
├── src/                         # Source code modules
│   ├── **init**.py
│   ├── data_loader.py           # Load raw MIT-BIH signals
│   ├── pipeline.py              # Window creation + preprocessing
│   ├── preprocessing.py         # Bandpass filter, normalization
│   ├── realtime_sim.py          # Real-time window streaming
│   ├── annotation_loader.py     # Load MIT-BIH annotations
│   ├── window_labeler.py        # Label windows (normal/abnormal)
├── dashboards/
│   ├── **init**.py
│   ├── test_ecg_dashboard.py    # Simulate live streaming dashboard (dummy alerts)
│   ├── live_dashboard_api.py    # Flask API to serve preprocessed windows
├── all_windows.npy              # Preprocessed 2-second windows
├── all_labels.npy               # Corresponding labels (0=normal, 1=abnormal)
├── test_ecg.py                  # Load dataset + sanity check plots
├── save_dataset.py              # Script used to generate .npy files (optional)
├── requirements.txt             # Python dependencies
└── README.md

````

---

## **2️⃣ Environment Setup**

1. Create virtual environment:

```bash
python -m venv ecg_env
````

2. Activate environment:

* **Windows PowerShell**

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser  # if needed
.\ecg_env\Scripts\activate
```

* **Linux / MacOS**

```bash
source ecg_env/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

> Dependencies include: `numpy`, `matplotlib`, `wfdb`, `flask`

---

## **3️⃣ How to Run**

### **A) Test the dataset**

```bash
python test_ecg.py
```

* Loads `all_windows.npy` and `all_labels.npy`
* Prints dataset stats
* Shows **sanity plots** of normal vs abnormal windows

---

### **B) Run live dashboard simulation**

```bash
python dashboards/test_ecg_dashboard.py
```

* Streams first 100 windows
* Highlights **abnormal windows** in red
* Uses **preprocessed dataset** → no need to process MIT-BIH files

---

### **C) Run Flask API (for frontend integration)**

```bash
python -m dashboards.live_dashboard_api
```

* Starts server at `http://127.0.0.1:5000`
* API endpoint to fetch window:

```
GET /get_window/<idx>
```

* Returns JSON:

```json
{
  "window_idx": 0,
  "label": 0,
  "data": [0.1, 0.2, ..., 0.0]
}
```

* Can be used to **stream windows in a web dashboard**

---

## **4️⃣ Dataset Notes**

* Windows are **2 seconds** (720 samples at 360 Hz)
* Stride / overlap: 50% → 360 samples
* Labels:

  * `0` → normal
  * `1` → abnormal
* No `None` values remain → dataset is clean

---

## **5️⃣ Recommended Next Steps for Person A**

1. Train **CNN / LSTM model** using `all_windows.npy` / `all_labels.npy`
2. Replace **dummy alert** in dashboards with **real predictions**
3. Evaluate metrics: accuracy, precision, recall, F1-score
4. Optional: enhance live API to serve **predictions + confidence %**

---

## **6️⃣ Important Notes**

* Only **one Flask API can run** at a time on port 5000
* Dashboards can either **use API** or **load `.npy` files directly**
* `.npy` files are already preprocessed → speeds up experimentation
* Make sure **MIT-BIH data** exists in `data/mitdb/` if any future raw processing is needed

---

**Prepared by Person B — Ready for integration**
