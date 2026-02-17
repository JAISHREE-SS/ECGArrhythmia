import numpy as np

def create_windows(signal, window_size=720, stride=360):
    windows = []
    max_start = len(signal) - window_size

    if max_start < 0:
        return np.empty((0, window_size, 1), dtype=float)

    for start in range(0, max_start + 1, stride):
        end = start + window_size
        window = signal[start:end]
        window = window.reshape(-1, 1)  # add channel dimension
        windows.append(window)

    return np.array(windows)

