import argparse
import os
import time

import wfdb
from wfdb.io import download as wfdb_download


def main():
    parser = argparse.ArgumentParser(description="Download INCART 12-lead Arrhythmia Database from PhysioNet.")
    parser.add_argument("--target-dir", default="data/incartdb", help="Local output directory")
    parser.add_argument("--version", default=None, help="Optional expected PhysioNet version")
    parser.add_argument("--retries", type=int, default=6, help="Retry attempts for transient network failures")
    parser.add_argument("--retry-wait-seconds", type=float, default=5.0, help="Base wait time between retries")
    parser.add_argument(
        "--records",
        default=None,
        help="Comma-separated record names to download, e.g. I75 or I01,I02,I75",
    )
    args = parser.parse_args()

    records = None
    if args.records:
        records = [rec.strip() for rec in args.records.split(",") if rec.strip()]

    os.makedirs(args.target_dir, exist_ok=True)

    db_name = "incartdb"
    resolved_version = wfdb_download.get_version(db_name)
    if args.version and args.version != resolved_version:
        raise ValueError(
            f"Requested version '{args.version}' does not match available version "
            f"'{resolved_version}' for wfdb.dl_database."
        )

    if records:
        print(
            f"Downloading {db_name}/{resolved_version} records={records} "
            f"to {args.target_dir} ..."
        )
    else:
        print(f"Downloading {db_name}/{resolved_version} to {args.target_dir} ...")

    # wfdb.dl_database can resume partial files when overwrite=False (default).
    last_error = None
    for attempt in range(1, args.retries + 1):
        try:
            wfdb.dl_database(
                db_name,
                dl_dir=args.target_dir,
                records=records if records is not None else "all",
                keep_subdirs=True,
            )
            print("Download complete.")
            return
        except Exception as exc:
            last_error = exc
            if attempt == args.retries:
                break
            sleep_s = args.retry_wait_seconds * (2 ** (attempt - 1))
            print(
                f"Attempt {attempt}/{args.retries} failed: {type(exc).__name__}: {exc}\n"
                f"Retrying in {sleep_s:.1f}s ..."
            )
            time.sleep(sleep_s)

    raise RuntimeError(
        f"Download failed after {args.retries} attempts. Last error: {type(last_error).__name__}: {last_error}"
    )


if __name__ == "__main__":
    main()
