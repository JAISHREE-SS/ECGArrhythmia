import argparse
import json
import os

import numpy as np


def _load_dataset(prefix):
    windows_path = f"{prefix}all_windows.npy"
    labels_path = f"{prefix}all_labels.npy"
    records_path = f"{prefix}all_record_names.npy"
    leads_path = f"{prefix}all_lead_names.npy"
    class_map_path = f"{prefix}class_map.json"

    required = [windows_path, labels_path, records_path, leads_path]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing files for prefix '{prefix}': {', '.join(missing)}"
        )

    windows = np.load(windows_path, allow_pickle=True)
    labels = np.load(labels_path, allow_pickle=True)
    record_names = np.load(records_path, allow_pickle=True)
    lead_names = np.load(leads_path, allow_pickle=True)

    if not (len(windows) == len(labels) == len(record_names) == len(lead_names)):
        raise ValueError(
            f"Length mismatch for prefix '{prefix}': "
            f"windows={len(windows)}, labels={len(labels)}, "
            f"records={len(record_names)}, leads={len(lead_names)}"
        )

    class_map = None
    if os.path.exists(class_map_path):
        with open(class_map_path, "r", encoding="utf-8") as f:
            class_map = json.load(f)

    return {
        "prefix": prefix,
        "windows": windows,
        "labels": labels,
        "record_names": record_names,
        "lead_names": lead_names,
        "class_map": class_map,
    }


def _record_set(record_names):
    return set(np.unique(record_names).astype(str).tolist())


def main():
    parser = argparse.ArgumentParser(
        description="Merge MIT and INCART datasets while avoiding duplicate records."
    )
    parser.add_argument(
        "--mit-prefix",
        default="mit_",
        help="Input prefix for MIT dataset files (e.g. 'mit_').",
    )
    parser.add_argument(
        "--incart-prefix",
        default="incart_",
        help="Input prefix for INCART dataset files (e.g. 'incart_').",
    )
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Output prefix for merged files. Use '' to write all_*.npy.",
    )
    parser.add_argument(
        "--on-duplicate-record",
        choices=["skip-second", "error"],
        default="skip-second",
        help="How to handle records appearing in both datasets.",
    )
    args = parser.parse_args()

    mit = _load_dataset(args.mit_prefix)
    incart = _load_dataset(args.incart_prefix)

    if (
        mit["class_map"] is not None
        and incart["class_map"] is not None
        and mit["class_map"] != incart["class_map"]
    ):
        raise ValueError("class_map.json mismatch between MIT and INCART datasets.")

    if mit["windows"].ndim != 3 or incart["windows"].ndim != 3:
        raise ValueError("Expected 3D windows arrays shaped like (N, T, C).")
    if mit["windows"].shape[1:] != incart["windows"].shape[1:]:
        raise ValueError(
            f"Window shape mismatch: MIT={mit['windows'].shape[1:]}, "
            f"INCART={incart['windows'].shape[1:]}"
        )

    mit_records = _record_set(mit["record_names"])
    incart_records = _record_set(incart["record_names"])
    duplicate_records = sorted(mit_records.intersection(incart_records))

    if duplicate_records and args.on_duplicate_record == "error":
        raise ValueError(
            "Duplicate record names detected across datasets: "
            + ", ".join(duplicate_records[:20])
        )

    if duplicate_records:
        incart_keep_mask = ~np.isin(incart["record_names"], duplicate_records)
    else:
        incart_keep_mask = np.ones(len(incart["record_names"]), dtype=bool)

    merged_windows = np.concatenate(
        [mit["windows"], incart["windows"][incart_keep_mask]], axis=0
    )
    merged_labels = np.concatenate(
        [mit["labels"], incart["labels"][incart_keep_mask]], axis=0
    )
    merged_records = np.concatenate(
        [mit["record_names"], incart["record_names"][incart_keep_mask]], axis=0
    )
    merged_leads = np.concatenate(
        [mit["lead_names"], incart["lead_names"][incart_keep_mask]], axis=0
    )

    out_windows = f"{args.output_prefix}all_windows.npy"
    out_labels = f"{args.output_prefix}all_labels.npy"
    out_records = f"{args.output_prefix}all_record_names.npy"
    out_leads = f"{args.output_prefix}all_lead_names.npy"
    out_map = f"{args.output_prefix}class_map.json"
    out_meta = f"{args.output_prefix}dataset_meta.json"

    np.save(out_windows, merged_windows)
    np.save(out_labels, merged_labels)
    np.save(out_records, merged_records)
    np.save(out_leads, merged_leads)

    class_map = mit["class_map"] if mit["class_map"] is not None else incart["class_map"]
    if class_map is not None:
        with open(out_map, "w", encoding="utf-8") as f:
            json.dump(class_map, f, indent=2)

    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_datasets": ["mitdb", "incartdb"],
                "mit_prefix": args.mit_prefix,
                "incart_prefix": args.incart_prefix,
                "output_prefix": args.output_prefix,
                "on_duplicate_record": args.on_duplicate_record,
                "duplicate_records_found": len(duplicate_records),
                "duplicate_records": duplicate_records,
                "total_windows": int(merged_windows.shape[0]),
                "window_shape": list(merged_windows.shape[1:]),
                "unique_records": int(len(np.unique(merged_records))),
                "unique_leads": int(len(np.unique(merged_leads))),
            },
            f,
            indent=2,
        )

    print("Merge complete")
    print(f"MIT windows: {len(mit['windows'])}")
    print(f"INCART windows: {len(incart['windows'])}")
    print(f"Dropped INCART windows due to duplicate records: {int(np.sum(~incart_keep_mask))}")
    print(f"Merged windows: {len(merged_windows)}")
    print(f"Window shape: {merged_windows.shape[1:]}")
    print(f"Unique records: {len(np.unique(merged_records))}")
    print(f"Unique leads: {len(np.unique(merged_leads))}")
    print(
        f"Saved: {out_windows}, {out_labels}, {out_records}, {out_leads}, "
        f"{out_map if class_map is not None else '(no class_map found in inputs)'}, {out_meta}"
    )


if __name__ == "__main__":
    main()
