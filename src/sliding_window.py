import numpy as np

def create_windows(signal, window_size=720, stride=360):
    windows = []

    for start in range(0, len(signal) - window_size, stride):
        end = start + window_size
        window = signal[start:end]
        window = window.reshape(-1, 1)  # add channel dimension
        windows.append(window)

    return np.array(windows)

