import numpy as np
from src.label_mapper import map_symbol_main_class

def label_windows(annotation_samples,
                  annotation_symbols,
                  signal_length,
                  window_size=720,
                  stride=360):

    window_labels = []
    max_start = signal_length - window_size

    if max_start < 0:
        return np.array(window_labels)

    for start in range(0, max_start + 1, stride):
        end = start + window_size

        labels_in_window = []

        for sample, symbol in zip(annotation_samples, annotation_symbols):
            if start <= sample < end:
                mapped = map_symbol_main_class(symbol)
                if mapped is not None:
                    labels_in_window.append(mapped)

        if len(labels_in_window) == 0:
            window_labels.append(None)
        else:
            present = set(int(label) for label in labels_in_window)
            # Presence-based priority so rare severe events are not hidden by normal majority.
            priority = [2, 3, 1, 4, 0]  # V > F > S > Q > N
            selected = next((label for label in priority if label in present), 0)
            window_labels.append(int(selected))


    return np.array(window_labels)
