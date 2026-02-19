import argparse
import datetime
import json
import os
import time
from math import gcd

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample_poly

from src.annotation_loader import load_annotations
from src.data_loader import get_record_list, load_record_all_leads
from src.pipeline import process_signal
from src.window_labeler import label_windows

CLASS_NAMES = {
    0: "N",
    1: "S",
    2: "V",
    3: "F",
    4: "Q",
}

def _as_int_hz(fs_value, source_name):
    fs_float = float(fs_value)
    fs_int = int(round(fs_float))
    if fs_int <= 0:
        raise ValueError(f"{source_name} must be positive. Got {fs_value}.")
    if not np.isclose(fs_float, fs_int):
        raise ValueError(
            f"{source_name} must be an integer-valued frequency for rational resampling. "
            f"Got {fs_value}."
        )
    return fs_int


def _resample_signals(signals, source_fs, target_fs):
    if source_fs == target_fs:
        return signals
    ratio_gcd = gcd(source_fs, target_fs)
    up = target_fs // ratio_gcd
    down = source_fs // ratio_gcd
    return resample_poly(signals, up=up, down=down, axis=0)


def _remap_annotation_samples(samples, source_fs, target_fs, target_signal_length):
    remapped = np.rint(np.asarray(samples, dtype=np.float64) * target_fs / source_fs).astype(np.int64)
    if target_signal_length <= 0:
        return remapped
    return np.clip(remapped, 0, target_signal_length - 1)


def main():
    parser = argparse.ArgumentParser(
        description="Build INCART 12-lead dataset in MIT-compatible output format."
    )
    parser.add_argument("--data-path", default="data/incartdb")
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--stride-seconds", type=float, default=1.0)
    parser.add_argument("--target-fs", type=float, default=360.0)
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Prefix for output files. Use '' for MIT-style names, or e.g. 'incart_'",
    )
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(args.data_path):
        raise FileNotFoundError(
            f"Dataset path not found: {args.data_path}\n"
            "Download INCART first, e.g.:\n"
            "python download_incart_data.py --target-dir data/incartdb"
        )

    records = sorted(get_record_list(args.data_path))
    if args.max_records is not None:
        records = records[: args.max_records]

    if len(records) == 0:
        raise RuntimeError(
            f"No WFDB records found in {args.data_path}. "
            "Expected .dat/.hea/.atr files for INCART records."
        )

    print(f"Total INCART records found: {len(records)}")

    all_windows = []
    all_labels = []
    all_record_names = []
    all_lead_names = []
    source_frequencies = set()

    target_fs = _as_int_hz(args.target_fs, "--target-fs")
    window_size = int(round(args.window_seconds * target_fs))
    stride = int(round(args.stride_seconds * target_fs))
    if window_size <= 0 or stride <= 0:
        raise ValueError("window-seconds and stride-seconds must produce positive sample counts.")

    for idx, record_name in enumerate(records):
        print(f"\nProcessing record {record_name} ({idx + 1}/{len(records)})")
        start_time = time.time()

        signals, fs, lead_names = load_record_all_leads(args.data_path, record_name)
        source_fs = _as_int_hz(fs, f"record {record_name} fs")
        source_frequencies.add(source_fs)
        signals = _resample_signals(signals, source_fs, target_fs)

        samples, symbols = load_annotations(args.data_path, record_name)
        remapped_samples = _remap_annotation_samples(
            samples,
            source_fs=source_fs,
            target_fs=target_fs,
            target_signal_length=signals.shape[0],
        )
        if np.any(remapped_samples[1:] < remapped_samples[:-1]):
            raise ValueError(f"Remapped annotation samples are not monotonic for record {record_name}.")

        labels = label_windows(
            remapped_samples,
            symbols,
            signal_length=signals.shape[0],
            window_size=window_size,
            stride=stride,
        )
        labels = np.array(labels, dtype=object)
        valid_idx = labels != None

        record_windows = 0
        for lead_idx, lead_name in enumerate(lead_names):
            signal = signals[:, lead_idx]
            windows = process_signal(signal, target_fs, window_size=window_size, stride=stride)

            if len(windows) != len(labels):
                raise ValueError(
                    f"Window/label mismatch for record={record_name}, lead={lead_name}: "
                    f"{len(windows)} windows vs {len(labels)} labels"
                )

            filtered_windows = windows[valid_idx]
            filtered_labels = labels[valid_idx]
            if len(filtered_windows) == 0:
                continue

            all_windows.append(filtered_windows.astype(np.float32, copy=False))
            all_labels.append(filtered_labels.astype(np.int16, copy=False))
            all_record_names.append(np.full(len(filtered_labels), record_name, dtype=object))
            all_lead_names.append(np.full(len(filtered_labels), lead_name, dtype=object))
            record_windows += len(filtered_windows)

        print(
            f"Done {record_name}: windows={record_windows}, leads={len(lead_names)}, "
            f"fs={source_fs}Hz -> target_fs={target_fs}Hz, "
            f"time={round(time.time() - start_time, 3)}s"
        )

    if len(all_windows) == 0:
        raise RuntimeError(
            "No valid windows were generated. Check INCART annotations and label mapping coverage."
        )

    all_windows = np.concatenate(all_windows, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_record_names = np.concatenate(all_record_names, axis=0)
    all_lead_names = np.concatenate(all_lead_names, axis=0)

    print("\nINCART processing complete")
    print(f"Total windows: {all_windows.shape[0]}")
    print(f"Window shape: {all_windows[0].shape}")
    print(f"Target fs: {target_fs}Hz")
    print(f"Unique records: {len(np.unique(all_record_names))}")
    print(f"Unique leads: {len(np.unique(all_lead_names))}")
    print(
        "Class counts -> "
        f"N={np.sum(all_labels == 0)}, "
        f"S={np.sum(all_labels == 1)}, "
        f"V={np.sum(all_labels == 2)}, "
        f"F={np.sum(all_labels == 3)}, "
        f"Q={np.sum(all_labels == 4)}"
    )

    out_windows = f"{args.output_prefix}all_windows.npy"
    out_labels = f"{args.output_prefix}all_labels.npy"
    out_records = f"{args.output_prefix}all_record_names.npy"
    out_leads = f"{args.output_prefix}all_lead_names.npy"
    out_map = f"{args.output_prefix}class_map.json"
    out_meta = f"{args.output_prefix}dataset_meta.json"

    np.save(out_windows, all_windows)
    np.save(out_labels, all_labels)
    np.save(out_records, all_record_names)
    np.save(out_leads, all_lead_names)
    with open(out_map, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in CLASS_NAMES.items()}, f, indent=2)
    source_fs_meta = sorted(source_frequencies)
    if len(source_fs_meta) == 1:
        source_fs_meta = source_fs_meta[0]
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_dataset": "incartdb",
                "source_fs": source_fs_meta,
                "target_fs": target_fs,
                "window_seconds": args.window_seconds,
                "stride_seconds": args.stride_seconds,
                "window_size_samples": window_size,
                "stride_samples": stride,
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
            f,
            indent=2,
        )

    print(f"Saved: {out_windows}, {out_labels}, {out_records}, {out_leads}, {out_map}, {out_meta}")

    if args.no_plot:
        return

    normal_candidates = np.where(all_labels == 0)[0]
    abnormal_candidates = np.where(all_labels != 0)[0]
    if len(normal_candidates) > 0 and len(abnormal_candidates) > 0:
        normal_idx = int(normal_candidates[0])
        abnormal_idx = int(abnormal_candidates[0])

        plt.figure(figsize=(10, 4))
        plt.subplot(2, 1, 1)
        plt.plot(all_windows[normal_idx].squeeze())
        plt.title(
            f"Sample {CLASS_NAMES[int(all_labels[normal_idx])]} | "
            f"Record={all_record_names[normal_idx]} | Lead={all_lead_names[normal_idx]}"
        )

        plt.subplot(2, 1, 2)
        plt.plot(all_windows[abnormal_idx].squeeze())
        plt.title(
            f"Sample {CLASS_NAMES[int(all_labels[abnormal_idx])]} | "
            f"Record={all_record_names[abnormal_idx]} | Lead={all_lead_names[abnormal_idx]}"
        )
        plt.tight_layout()
        plt.show()
    else:
        print("Skipped plotting: normal or abnormal class not found.")


if __name__ == "__main__":
    main()
