import time
from src.sliding_window import create_windows

def realtime_stream(signal, fs, window_size=720, stride=360, delay=1):
    windows = create_windows(signal, window_size, stride)

    for i, window in enumerate(windows):
        print(f"Streaming window {i+1}/{len(windows)}")
        yield window
        time.sleep(delay)  # simulate real-time delay (1 sec)
