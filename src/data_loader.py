import os
import wfdb

def get_record_list(data_path):
    records = []
    for file in os.listdir(data_path):
        if file.endswith(".dat"):
            records.append(file.replace(".dat", ""))
    return records


def load_record(data_path, record_name):
    record_path = os.path.join(data_path, record_name)
    record = wfdb.rdrecord(record_path)
    signal = record.p_signal[:, 0]  # MLII only
    fs = record.fs
    return signal, fs
