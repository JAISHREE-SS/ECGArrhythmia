import wfdb
import os

def load_annotations(data_path, record_name):
    record_path = os.path.join(data_path, record_name)
    annotation = wfdb.rdann(record_path, 'atr')
    return annotation.sample, annotation.symbol
