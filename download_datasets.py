#!/usr/bin/env python3
"""
download_datasets.py

Download full Hugging Face dataset repos to local disk, one folder per
dataset_id, so you can parse them locally afterward instead of fighting
partial/streaming access.

Uses huggingface_hub.snapshot_download, which:
  - downloads every file in the repo (parquet, loose images, zips, README, etc.)
  - resumes partial downloads automatically (safe to Ctrl+C and rerun)
  - skips repos already fully downloaded on a rerun (checked via a marker file)

TIMEOUT WATCHDOG: snapshot_download has no built-in wall-clock timeout on
the actual file transfer - a stalled connection (as opposed to one that
errors outright) can hang indefinitely. Each dataset's download now runs
in a worker thread with a hard --timeout (default 300s); if it's
exceeded, that dataset is abandoned for this run (its thread may keep
running in the background, same tradeoff as list_parquet_sizes.py) and
NOT marked done, so a rerun retries it - any partial files it already
wrote are kept and resumed via snapshot_download's own resume support.

Usage:
    python download_datasets.py --input dataset_parquet.csv --out-dir ./raw_datasets

    # only download rows still needing it (e.g. skip anything already marked
    # image_confirmed in a previous classification run, if you only care
    # about the ones needing manual review):
    python download_datasets.py --input results.csv --out-dir ./raw_datasets \
        --status-column status --only-status needs_user_review

    # bump the per-dataset timeout for big/slow repos:
    python download_datasets.py --input results.csv --out-dir ./raw_datasets \
        --timeout 900

Requires:
    pip install --break-system-packages huggingface_hub pandas
"""

import argparse
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import pandas as pd
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError

HF_TOKEN = os.environ.get("HF_TOKEN")

DONE_MARKER = ".download_complete"

# huggingface_hub reads this to bound its own metadata/connect timeouts;
# the --timeout watchdog below is the real backstop for stalled transfers.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")


def already_done(dest_dir):
    return os.path.exists(os.path.join(dest_dir, DONE_MARKER))


def mark_done(dest_dir):
    with open(os.path.join(dest_dir, DONE_MARKER), "w") as f:
        f.write(str(time.time()))


def _snapshot_download_once(dataset_id, dest_dir):
    snapshot_download(
        repo_id=dataset_id,
        repo_type="dataset",
        local_dir=dest_dir,
        token=HF_TOKEN,
        # keep going even if one file 404s/500s inside an otherwise
        # fine repo, rather than aborting the whole snapshot
        max_workers=4,
    )


def download_one(dataset_id, out_dir, executor_holder, timeout, max_retries=3):
    """executor_holder is a 1-element list holding the current
    ThreadPoolExecutor, so a timed-out one can be swapped out by the
    caller without this function needing to manage executor lifecycle."""
    safe_name = dataset_id.replace("/", "__")
    dest_dir = os.path.join(out_dir, safe_name)

    if already_done(dest_dir):
        return "already_done", None

    last_err = None
    for attempt in range(1, max_retries + 1):
        future = executor_holder[0].submit(_snapshot_download_once, dataset_id, dest_dir)
        try:
            future.result(timeout=timeout)
            mark_done(dest_dir)
            return "ok", None
        except FutureTimeoutError:
            last_err = f"timed_out_after_{timeout}s"
            # Abandon the stuck future's thread and get a fresh executor so
            # the next dataset (or retry) isn't queued behind a hung worker.
            executor_holder[0].shutdown(wait=False, cancel_futures=True)
            executor_holder[0] = ThreadPoolExecutor(max_workers=1)
            if attempt < max_retries:
                print(f"[retry {attempt}/{max_retries} after timeout] ", end="", flush=True)
                time.sleep(3 * attempt)
        except HfHubHTTPError as e:
            status = getattr(e.response, "status_code", None)
            last_err = f"HTTP {status}: {e}"
            if status in (401, 403):
                # gated/private and no access — retrying won't help
                return "failed", last_err
            if attempt < max_retries:
                time.sleep(3 * attempt)
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < max_retries:
                time.sleep(3 * attempt)

    return "failed", last_err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV with a dataset_id column")
    parser.add_argument("--out-dir", required=True, help="Where to save each repo (one subfolder per dataset)")
    parser.add_argument("--id-column", default="dataset_id")
    parser.add_argument("--status-column", default=None,
                         help="Optional column to filter on (e.g. 'status' from a prior classification run)")
    parser.add_argument("--only-status", nargs="*", default=None,
                         help="If --status-column is set, only download rows whose value is one of these")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=300.0,
                         help="Hard wall-clock timeout in seconds per dataset download (default 300). "
                              "A stalled (not erroring) connection can otherwise hang forever. Raise this "
                              "for large repos on slow connections rather than lowering it aggressively.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.id_column not in df.columns:
        raise SystemExit(f"Column '{args.id_column}' not found. Available: {list(df.columns)}")

    if args.status_column and args.only_status:
        if args.status_column not in df.columns:
            raise SystemExit(f"Column '{args.status_column}' not found. Available: {list(df.columns)}")
        df = df[df[args.status_column].isin(args.only_status)]

    ids = df[args.id_column].dropna().unique().tolist()
    if args.limit:
        ids = ids[: args.limit]

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Downloading {len(ids)} dataset repo(s) into {os.path.abspath(args.out_dir)} "
          f"(per-dataset timeout: {args.timeout:.0f}s) ...\n")
    if not HF_TOKEN:
        print("NOTE: no HF_TOKEN set. Unauthenticated requests get a much lower rate limit "
              "and are more likely to stall. Set HF_TOKEN to raise that ceiling.\n")

    results = {"ok": 0, "already_done": 0, "failed": 0}
    failures = []

    # Single-worker pool acts as the timeout watchdog, same pattern as
    # list_parquet_sizes.py: submit(), then future.result(timeout=...).
    executor_holder = [ThreadPoolExecutor(max_workers=1)]

    for i, dataset_id in enumerate(ids):
        dataset_id = str(dataset_id).strip()
        if not dataset_id:
            continue
        print(f"[{i+1}/{len(ids)}] {dataset_id} ...", end=" ", flush=True)
        t0 = time.time()
        try:
            status, err = download_one(dataset_id, args.out_dir, executor_holder, args.timeout)
        except KeyboardInterrupt:
            print("\nInterrupted by user. Progress so far is saved — rerun the same command to resume.")
            break
        except Exception:
            status, err = "failed", traceback.format_exc(limit=1)

        results[status] = results.get(status, 0) + 1
        if status == "ok":
            print(f"done [{time.time()-t0:.1f}s]")
        elif status == "already_done":
            print("already downloaded, skipping")
        else:
            print(f"FAILED [{time.time()-t0:.1f}s]: {err}")
            failures.append((dataset_id, err))

    executor_holder[0].shutdown(wait=False, cancel_futures=True)

    print(f"\nSummary: {results['ok']} downloaded, {results['already_done']} already done, "
          f"{results.get('failed', 0)} failed.")
    if failures:
        print("\nFailed datasets (rerun the same command to retry — partial files resume):")
        for did, err in failures:
            print(f"  - {did}: {err}")


if __name__ == "__main__":
    main()