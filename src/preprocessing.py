import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(signal, fs, lowcut=0.5, highcut=40, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    # Keep digital filter bounds valid and fail clearly for invalid inputs.
    if low <= 0 or low >= 1:
        raise ValueError(f"Invalid lowcut/fs combination: low={low:.6f}.")
    if high <= 0:
        raise ValueError(f"Invalid highcut/fs combination: high={high:.6f}.")
    if high >= 1:
        high = 0.99
    if low >= high:
        raise ValueError(
            f"Invalid bandpass range after normalization: low={low:.6f}, high={high:.6f}."
        )

    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


def normalize_signal(signal):
    std = np.std(signal)
    if std == 0:
        # Return zeros for flat signals to avoid NaN/Inf propagation.
        return np.zeros_like(signal)
    return (signal - np.mean(signal)) / std
