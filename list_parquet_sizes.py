#!/usr/bin/env python3
"""
list_parquet_sizes.py
For each dataset ID in an input CSV, list all .parquet files in that HF
dataset repo along with their sizes, and write two output CSVs:

  1. a detailed file, one row per parquet file
     (dataset_id, parquet_file, size_bytes, size_mb)
  2. a per-dataset summary file
     (dataset_id, parquet_count, total_size_bytes, total_size_mb, error)

Only metadata is fetched (file listing + sizes) - no parquet content is
ever downloaded.

Resilience:
  - Each dataset is given a hard wall-clock timeout (--timeout, default
    30s). A hung/slow request no longer stalls the whole run - it's
    logged as an error and the run moves on.
  - Results are written incrementally, one dataset at a time, and flushed
    immediately - Ctrl-C or a crash loses at most the in-flight dataset.
  - Re-running with the same --summary-out automatically resumes: any
    dataset_id already present there is skipped.

Usage:
    python list_parquet_sizes.py image_urls.csv --id-column "dataset_id" \
        --detail-out parquet_files.csv --summary-out parquet_summary.csv
"""

import argparse
import csv
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import pandas as pd
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError, HfHubHTTPError

HF_TOKEN = os.environ.get("HF_TOKEN")
api = HfApi(token=HF_TOKEN)

DETAIL_FIELDS = ["dataset_id", "parquet_file", "size_bytes", "size_mb"]
SUMMARY_FIELDS = ["dataset_id", "parquet_count", "total_size_bytes", "total_size_mb", "error"]

# Small pause between every dataset, purely to avoid hammering the API -
# HF's unauthenticated rate limit is 500 requests/300s, and a repo with
# many files can burn several requests (pagination) in one go.
MIN_REQUEST_INTERVAL = 0.3

DEFAULT_RETRY_AFTER = 130.0  # fallback if we can't parse one from the error
MAX_RATE_LIMIT_RETRIES = 5


def _extract_retry_after(exc) -> float:
    """Best-effort extraction of a Retry-After duration (seconds) from a
    429 error, whether it's exposed via response headers or only in the
    error message text."""
    response = getattr(exc, "response", None)
    if response is not None:
        val = response.headers.get("Retry-After") or response.headers.get("retry-after")
        if val:
            try:
                return float(val)
            except ValueError:
                pass
    match = re.search(r"Retry after (\d+(?:\.\d+)?) seconds", str(exc))
    if match:
        return float(match.group(1))
    return DEFAULT_RETRY_AFTER


def _is_rate_limit_error(exc) -> bool:
    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", None) == 429:
        return True
    text = str(exc)
    return "429" in text or "rate limit" in text.lower()


def get_parquet_files_with_sizes(repo_id: str):
    """Returns a list of (path, size_bytes) for every .parquet file in the
    dataset repo. Runs in a worker thread so it can be timed out from the
    caller - huggingface_hub doesn't expose a reliable per-call timeout
    for list_repo_files/get_paths_info in all versions, so a wall-clock
    watchdog is the robust way to bound it."""
    files = api.list_repo_files(repo_id, repo_type="dataset")
    parquet_paths = [f for f in files if f.lower().endswith(".parquet")]
    if not parquet_paths:
        return []
    infos = api.get_paths_info(repo_id, parquet_paths, repo_type="dataset")
    results = []
    for info in infos:
        size = getattr(info, "size", None)
        results.append((info.path, size if size is not None else 0))
    return results


def load_done_ids(summary_path):
    done = set()
    if os.path.exists(summary_path):
        with open(summary_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("dataset_id"):
                    done.add(row["dataset_id"].strip())
    return done


def ensure_header(path, fieldnames):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def append_rows(path, fieldnames, rows):
    if not rows:
        return
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def main():
    parser = argparse.ArgumentParser(description="List parquet files and sizes per HF dataset.")
    parser.add_argument("csv_file", help="Path to input CSV containing dataset IDs")
    parser.add_argument("--id-column", default="Dataset ID", help="Column name containing dataset repo IDs")
    parser.add_argument("--detail-out", default="parquet_files.csv", help="Output CSV: one row per parquet file")
    parser.add_argument("--summary-out", default="parquet_summary.csv", help="Output CSV: one row per dataset")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of datasets processed")
    parser.add_argument("--timeout", type=float, default=25.0,
                         help="Per-dataset wall-clock timeout in seconds (default 25). "
                              "Keep this generous - a metadata listing call for a repo with many "
                              "files takes several paginated requests, so a couple of seconds is "
                              "not a real hang, just normal latency. Setting this too low causes "
                              "false timeouts, which leak background threads still retrying against "
                              "the API - that's what triggers rate-limit storms.")
    args = parser.parse_args()

    if not os.path.exists(args.csv_file):
        print(f"Error: input file not found: {args.csv_file}")
        sys.exit(1)

    df = pd.read_csv(args.csv_file)
    if args.id_column not in df.columns:
        print(f"Error: column '{args.id_column}' not found. Available: {list(df.columns)}")
        sys.exit(1)

    repo_ids = list(df[args.id_column].dropna().unique())
    if args.limit:
        repo_ids = repo_ids[: args.limit]

    if args.timeout < 10:
        print(f"WARNING: --timeout={args.timeout}s is very aggressive for a metadata call. "
              f"Timeouts that aren't real hangs leak background threads that keep hitting the "
              f"API - this is what caused the 429 rate-limit storm last run. Consider 20-30s+.")

    if not HF_TOKEN:
        print("NOTE: no HF_TOKEN set. Unauthenticated requests get a much lower rate limit "
              "(500 req/300s). Set HF_TOKEN to raise that ceiling and reduce 429s.")

    ensure_header(args.detail_out, DETAIL_FIELDS)
    ensure_header(args.summary_out, SUMMARY_FIELDS)
    done_ids = load_done_ids(args.summary_out)
    if done_ids:
        print(f"Resuming: {len(done_ids)} datasets already done, skipping those.")

    # A single-worker pool acts as our timeout watchdog: submit(), then
    # future.result(timeout=...). If it times out we abandon that future -
    # its thread may keep running in the background - and spin up a fresh
    # worker so the next dataset isn't queued behind it. With a sane
    # timeout this should be rare.
    executor = ThreadPoolExecutor(max_workers=1)

    total = len(repo_ids)
    processed = 0
    try:
        for i, repo_id in enumerate(repo_ids):
            repo_id = str(repo_id).strip()
            if repo_id in done_ids:
                continue

            print(f"[{i+1}/{total}] {repo_id}...", end=" ", flush=True)
            time.sleep(MIN_REQUEST_INTERVAL)

            rate_limit_attempts = 0
            while True:
                future = executor.submit(get_parquet_files_with_sizes, repo_id)
                try:
                    files_with_sizes = future.result(timeout=args.timeout)
                    break
                except FutureTimeoutError:
                    print(f"timed out after {args.timeout}s, skipping")
                    append_rows(args.summary_out, SUMMARY_FIELDS, [{
                        "dataset_id": repo_id, "parquet_count": "NA", "total_size_bytes": "NA",
                        "total_size_mb": "NA", "error": f"timeout_after_{args.timeout}s",
                    }])
                    executor.shutdown(wait=False, cancel_futures=True)
                    executor = ThreadPoolExecutor(max_workers=1)
                    files_with_sizes = None
                    break
                except (HfHubHTTPError, Exception) as e:
                    if isinstance(e, GatedRepoError):
                        print("gated/no access, skipping")
                        append_rows(args.summary_out, SUMMARY_FIELDS, [{
                            "dataset_id": repo_id, "parquet_count": "NA", "total_size_bytes": "NA",
                            "total_size_mb": "NA", "error": "gated_or_no_access",
                        }])
                        files_with_sizes = None
                        break
                    if isinstance(e, RepositoryNotFoundError):
                        print("not found, skipping")
                        append_rows(args.summary_out, SUMMARY_FIELDS, [{
                            "dataset_id": repo_id, "parquet_count": "NA", "total_size_bytes": "NA",
                            "total_size_mb": "NA", "error": "repo_not_found",
                        }])
                        files_with_sizes = None
                        break
                    if _is_rate_limit_error(e):
                        rate_limit_attempts += 1
                        if rate_limit_attempts > MAX_RATE_LIMIT_RETRIES:
                            print(f"still rate-limited after {MAX_RATE_LIMIT_RETRIES} retries, "
                                  f"giving up on this dataset for now (will retry next run)")
                            files_with_sizes = "RATE_LIMIT_GIVE_UP"
                            break
                        wait_s = _extract_retry_after(e)
                        print(f"rate limited, waiting {wait_s:.0f}s (retry {rate_limit_attempts}/{MAX_RATE_LIMIT_RETRIES})...",
                              end=" ", flush=True)
                        time.sleep(wait_s)
                        print("retrying...", end=" ", flush=True)
                        continue
                    print(f"error: {e}")
                    append_rows(args.summary_out, SUMMARY_FIELDS, [{
                        "dataset_id": repo_id, "parquet_count": "NA", "total_size_bytes": "NA",
                        "total_size_mb": "NA", "error": str(e),
                    }])
                    files_with_sizes = None
                    break

            if files_with_sizes == "RATE_LIMIT_GIVE_UP":
                # Deliberately do NOT write a summary row - this dataset
                # stays un-"done" so a re-run retries it instead of
                # permanently skipping it.
                continue
            if files_with_sizes is None:
                processed += 1
                continue

            if not files_with_sizes:
                print("no parquet files")
                append_rows(args.summary_out, SUMMARY_FIELDS, [{
                    "dataset_id": repo_id, "parquet_count": 0, "total_size_bytes": 0,
                    "total_size_mb": 0.0, "error": "",
                }])
                processed += 1
                continue

            total_bytes = sum(size for _, size in files_with_sizes)
            detail_rows = [{
                "dataset_id": repo_id,
                "parquet_file": path,
                "size_bytes": size,
                "size_mb": round(size / (1024 * 1024), 2),
            } for path, size in files_with_sizes]

            append_rows(args.detail_out, DETAIL_FIELDS, detail_rows)
            append_rows(args.summary_out, SUMMARY_FIELDS, [{
                "dataset_id": repo_id,
                "parquet_count": len(files_with_sizes),
                "total_size_bytes": total_bytes,
                "total_size_mb": round(total_bytes / (1024 * 1024), 2),
                "error": "",
            }])
            print(f"{len(files_with_sizes)} files, {total_bytes / (1024*1024):.1f} MB total")
            processed += 1
    except KeyboardInterrupt:
        print(f"\nInterrupted. {processed} datasets processed this run were already written to disk. "
              f"Re-run the same command to resume from here.")
        sys.exit(1)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    print(f"\nDone. Processed {processed} datasets this run.")
    print(f"Detail CSV: {args.detail_out}")
    print(f"Summary CSV: {args.summary_out}")


if __name__ == "__main__":
    main()