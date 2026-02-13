import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(signal, fs, lowcut=0.5, highcut=40, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


def normalize_signal(signal):
    return (signal - np.mean(signal)) / np.std(signal)
