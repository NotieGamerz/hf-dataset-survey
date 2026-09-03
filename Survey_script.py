#!/usr/bin/env python3
"""
Survey_script.py

Single entrypoint that chains the whole HF dataset survey pipeline:

  1. fetch_query_metadata_from_huggingface.py <query>
       -> hf_<query>_candidates/hf_<query>_candidates.csv
  2. dataset_cleaner.py  (splits/dedupes that CSV)
       -> filtered_hf_<query>_candidates_main.csv (+ has_parquet.csv etc.)
  3. validate_datasets.py  (VLM classification), run on the deduped main.csv
       -> classified_<query>_dataset.csv
  4. (only if --download-parquet was passed)
       list_parquet_sizes.py on the has_parquet.csv split -> total download size
       compare against free disk space, ask for y/n confirmation
       -> if yes: download_datasets.py on the has_parquet.csv split

Every stage's stdout/stderr is streamed live to the terminal AND written to
its own timestamped log file under ./logs/<query>_<run_ts>/.

Usage:
    python Survey_script.py uav \\
        --vlm-base-url http://127.0.0.1:8000/v1 \\
        --vlm-model lmstudio-community/Qwen3-VL-8B-Instruct-MLX-4bit \\
        --download-parquet

All five worker scripts (fetch_query_metadata_from_huggingface.py,
dataset_cleaner.py, validate_datasets.py, list_parquet_sizes.py,
download_datasets.py) must sit next to this file.
"""

import argparse
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

FETCH_SCRIPT    = SCRIPT_DIR / "fetch_query_metadata_from_huggingface.py"
CLEANER_SCRIPT  = SCRIPT_DIR / "dataset_cleaner.py"
VALIDATE_SCRIPT = SCRIPT_DIR / "validate_datasets.py"
SIZES_SCRIPT    = SCRIPT_DIR / "list_parquet_sizes.py"
DOWNLOAD_SCRIPT = SCRIPT_DIR / "download_datasets.py"

REQUIRED_SCRIPTS = [FETCH_SCRIPT, CLEANER_SCRIPT, VALIDATE_SCRIPT, SIZES_SCRIPT, DOWNLOAD_SCRIPT]


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def slugify(query: str) -> str:
    """Mirrors get_output_paths()'s slug logic in
    fetch_query_metadata_from_huggingface.py, so we can predict the output
    folder/file names without parsing that script's stdout."""
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in query).strip("_").lower()
    return safe or "dataset"


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"


def run_stage(title: str, cmd: list, log_path: Path, cwd: Path = None) -> int:
    """Runs a subprocess, streaming its combined stdout/stderr line-by-line
    to both the terminal and a log file simultaneously. Returns the exit
    code. Raises SystemExit if the stage fails."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    banner = f"\n{'=' * 78}\nSTAGE: {title}\n{'=' * 78}\n$ {' '.join(shlex.quote(c) for c in cmd)}\n"
    print(banner, flush=True)

    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write(banner)
        logf.flush()

        start = time.time()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd or SCRIPT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            logf.write(line)
        proc.wait()
        elapsed = time.time() - start

        footer = f"\n[{title}] exited with code {proc.returncode} after {elapsed:.1f}s\n"
        print(footer, flush=True)
        logf.write(footer)

    if proc.returncode != 0:
        print(f"\n!! Stage '{title}' failed (exit {proc.returncode}). Full log: {log_path}")
        raise SystemExit(proc.returncode)

    print(f"[{title}] log saved to: {log_path}")
    return proc.returncode


def confirm_yes_no(prompt: str) -> bool:
    while True:
        ans = input(f"{prompt} [y/n]: ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("Please answer 'y' or 'n'.")


def parse_passthrough(raw: str) -> list:
    """Splits a raw extra-args string (e.g. '--n-images 50 --limit 5')
    the way a shell would, respecting quotes."""
    if not raw:
        return []
    return shlex.split(raw)


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run the full HF dataset survey pipeline end-to-end.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("query", help="Search query, e.g. 'uav', 'drone'")

    # --- fetch stage ---
    parser.add_argument("--sort", default=None, help="Passed through to fetch script's --sort (default: trendingScore)")
    parser.add_argument("--fetch-limit", type=int, default=None, help="Passed through to fetch script's --limit (max datasets to retrieve)")
    parser.add_argument("--fetch-tags", default="", help="Extra raw CLI args appended verbatim to the fetch stage, e.g. --fetch-tags '--output-dir custom_dir'")

    # --- clean stage ---
    parser.add_argument("--clean-tags", default="", help="Extra raw CLI args appended verbatim to the dataset_cleaner.py stage")

    # --- validate/classify stage ---
    parser.add_argument("--vlm-base-url", default=None, help="Base URL for the image VLM backend (required unless --skip-validate)")
    parser.add_argument("--vlm-model", default=None, help="Model name for the image VLM backend (required unless --skip-validate)")
    parser.add_argument("--vlm-api-key", default=None, help="API key for the VLM backend, if needed")
    parser.add_argument("--classify-tags", default="", help="Extra raw CLI args appended verbatim to validate_datasets.py, e.g. --classify-tags '--n-images 50 --limit 5'")
    parser.add_argument("--skip-validate", action="store_true", help="Skip the VLM classification stage entirely")

    # --- parquet download stage ---
    parser.add_argument("--download-parquet", action="store_true", help="After validation, size-check and (on confirmation) download the has_parquet split via download_datasets.py")
    parser.add_argument("--download-dir", default=None, help="Where downloaded dataset repos go (default: ./raw_datasets)")
    parser.add_argument("--sizes-timeout", type=float, default=25.0, help="Per-dataset timeout passed to list_parquet_sizes.py (default 25s)")
    parser.add_argument("--download-tags", default="", help="Extra raw CLI args appended verbatim to download_datasets.py")
    parser.add_argument("--assume-yes", action="store_true", help="Skip the disk-space confirmation prompt and proceed automatically")

    args = parser.parse_args()

    missing = [str(p.name) for p in REQUIRED_SCRIPTS if not p.exists()]
    if missing:
        print(f"ERROR: expected these scripts next to Survey_script.py but they're missing: {missing}")
        raise SystemExit(1)

    if not args.skip_validate and (not args.vlm_base_url or not args.vlm_model):
        print("ERROR: --vlm-base-url and --vlm-model are required for the validate stage "
              "(or pass --skip-validate to skip it).")
        raise SystemExit(1)

    slug = slugify(args.query)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = SCRIPT_DIR / "logs" / f"{slug}_{run_ts}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run logs: {log_dir}\n")

    candidates_dir = SCRIPT_DIR / f"hf_{slug}_candidates"
    candidates_csv = candidates_dir / f"hf_{slug}_candidates.csv"

    # ── Stage 1: fetch ────────────────────────────────────────────────────
    fetch_cmd = [sys.executable, str(FETCH_SCRIPT), args.query]
    if args.sort:
        fetch_cmd += ["--sort", args.sort]
    if args.fetch_limit is not None:
        fetch_cmd += ["--limit", str(args.fetch_limit)]
    fetch_cmd += parse_passthrough(args.fetch_tags)
    run_stage("1/4 Fetch metadata", fetch_cmd, log_dir / "01_fetch.log")

    if not candidates_csv.exists():
        print(f"ERROR: expected fetch output at {candidates_csv} but it wasn't created. "
              f"If you passed --output-dir via --fetch-tags, also pass matching --clean-tags '--output-dir ...' etc.")
        raise SystemExit(1)

    # ── Stage 2: clean/split/dedupe ──────────────────────────────────────
    clean_cmd = [sys.executable, str(CLEANER_SCRIPT), "--input", str(candidates_csv)]
    clean_cmd += parse_passthrough(args.clean_tags)
    run_stage("2/4 Clean & split", clean_cmd, log_dir / "02_clean.log")

    main_csv = candidates_dir / f"filtered_hf_{slug}_candidates_main.csv"
    has_parquet_csv = candidates_dir / f"filtered_hf_{slug}_candidates_has_parquet.csv"

    if not main_csv.exists():
        print(f"NOTE: {main_csv} was not created (no included datasets after dedupe?). "
              f"Skipping validate/download stages.")
        return

    # ── Stage 3: VLM classification ──────────────────────────────────────
    if not args.skip_validate:
        classified_csv = SCRIPT_DIR / f"classified_{slug}_dataset.csv"
        classify_cmd = [
            sys.executable, str(VALIDATE_SCRIPT),
            "--input", str(main_csv),
            "--output", str(classified_csv),
            "--vlm-base-url", args.vlm_base_url,
            "--vlm-model", args.vlm_model,
        ]
        if args.vlm_api_key:
            classify_cmd += ["--vlm-api-key", args.vlm_api_key]
        classify_cmd += parse_passthrough(args.classify_tags)
        run_stage("3/4 VLM validation", classify_cmd, log_dir / "03_validate.log")
    else:
        print("Skipping validate stage (--skip-validate).")

    # ── Stage 4: optional parquet download, gated by disk-space check ────
    if args.download_parquet:
        if not has_parquet_csv.exists():
            print(f"NOTE: {has_parquet_csv} doesn't exist (no parquet datasets found). Skipping download stage.")
            return

        sizes_dir = candidates_dir / "parquet_sizes"
        sizes_dir.mkdir(parents=True, exist_ok=True)
        detail_out = sizes_dir / f"{slug}_parquet_files.csv"
        summary_out = sizes_dir / f"{slug}_parquet_summary.csv"

        sizes_cmd = [
            sys.executable, str(SIZES_SCRIPT), str(has_parquet_csv),
            "--id-column", "Dataset ID",
            "--detail-out", str(detail_out),
            "--summary-out", str(summary_out),
            "--timeout", str(args.sizes_timeout),
        ]
        run_stage("4/4a Size-check parquet", sizes_cmd, log_dir / "04a_sizecheck.log")

        # Sum every row in the summary CSV (NA/errored rows contribute 0).
        import csv as _csv
        total_bytes = 0
        n_rows = 0
        n_errored = 0
        if summary_out.exists():
            with open(summary_out, newline="", encoding="utf-8") as f:
                for row in _csv.DictReader(f):
                    n_rows += 1
                    raw = row.get("total_size_bytes", "")
                    try:
                        total_bytes += int(float(raw))
                    except (TypeError, ValueError):
                        n_errored += 1

        free_bytes = shutil.disk_usage(SCRIPT_DIR).free

        print(f"\nParquet size check ({n_rows} datasets, {n_errored} with no size data):")
        print(f"  Total download size : {human_bytes(total_bytes)}")
        print(f"  Free disk space now  : {human_bytes(free_bytes)}")
        if total_bytes > free_bytes:
            print(f"  !! WARNING: download size exceeds free disk space by {human_bytes(total_bytes - free_bytes)}.")
        else:
            print(f"  OK: {human_bytes(free_bytes - total_bytes)} of headroom would remain after download.")

        proceed = args.assume_yes or confirm_yes_no("\nProceed with downloading these parquet datasets?")
        if not proceed:
            print("Download cancelled by user. Pipeline finished (fetch/clean/validate results are still on disk).")
            return

        download_dir = Path(args.download_dir) if args.download_dir else SCRIPT_DIR / "raw_datasets"
        download_cmd = [
            sys.executable, str(DOWNLOAD_SCRIPT),
            "--input", str(has_parquet_csv),
            "--out-dir", str(download_dir),
            "--id-column", "Dataset ID",
        ]
        download_cmd += parse_passthrough(args.download_tags)
        run_stage("4/4b Download parquet datasets", download_cmd, log_dir / "04b_download.log")

    print(f"\nAll requested stages complete. Logs: {log_dir}")


if __name__ == "__main__":
    main()