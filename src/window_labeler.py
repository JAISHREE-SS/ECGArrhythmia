import numpy as np
from src.label_mapper import map_symbol

def label_windows(annotation_samples,
                  annotation_symbols,
                  signal_length,
                  window_size=720,
                  stride=360):

    window_labels = []

    for start in range(0, signal_length - window_size, stride):
        end = start + window_size

        labels_in_window = []

        for sample, symbol in zip(annotation_samples, annotation_symbols):
            if start <= sample < end:
                mapped = map_symbol(symbol)
                if mapped is not None:
                    labels_in_window.append(mapped)

        if len(labels_in_window) == 0:
            window_labels.append(None)
        else:
            if 1 in labels_in_window:      # Abnormal present
                window_labels.append(1)
            elif 0 in labels_in_window:    # Only normal
                window_labels.append(0)
            else:
                window_labels.append(None)


    return np.array(window_labels)
