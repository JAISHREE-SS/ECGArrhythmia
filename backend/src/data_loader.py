import os
import wfdb

def get_record_list(data_path):
    records = []
    for file in os.listdir(data_path):
        if file.endswith(".dat"):
            records.append(file.replace(".dat", ""))
    return records


def load_record(data_path, record_name, lead_index=0):
    record_path = os.path.join(data_path, record_name)
    record = wfdb.rdrecord(record_path)
    signal = record.p_signal[:, lead_index]
    fs = record.fs
    return signal, fs


def load_record_all_leads(data_path, record_name):
    record_path = os.path.join(data_path, record_name)
    record = wfdb.rdrecord(record_path)
    signals = record.p_signal
    fs = record.fs
    lead_names = list(record.sig_name) if record.sig_name is not None else [
        f"lead_{idx}" for idx in range(signals.shape[1])
    ]
    return signals, fs, lead_names
