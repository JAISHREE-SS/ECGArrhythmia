from src.preprocessing import bandpass_filter, normalize_signal
from src.sliding_window import create_windows

def process_signal(signal, fs, window_size=720, stride=360):
    filtered = bandpass_filter(signal, fs)
    normalized = normalize_signal(filtered)
    windows = create_windows(normalized, window_size, stride)
    return windows
