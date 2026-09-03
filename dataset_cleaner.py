#!/usr/bin/env python3
"""
dataset_cleaner.py

Takes the CSV produced by fetch_query_metadata_from_huggingface.py and
produces a deduplicated main CSV of every dataset that contains at least
one image, video, or parquet file, then splits it into task-specific CSVs.

KEY BEHAVIORS:
1. NO CACHES: Each dataset is written directly to CSVs as soon as it is evaluated.
   If interrupted, re-running reads the existing CSVs to resume where it left off.
2. TIMEOUT (5 seconds): Any repo that takes > 5 seconds or fails is skipped/excluded.
3. PARQUET IMAGE DETECTION: Queries the HF Dataset Viewer /info endpoint (5s timeout)
   without downloading datasets.
"""

import argparse
import csv
import difflib
import os
import re
import socket
import time
from collections import Counter

import pandas as pd
import requests

try:
    # pyrefly: ignore [missing-import]
    from huggingface_hub import HfApi
    # pyrefly: ignore [missing-import]
    from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError, HfHubHTTPError
except ImportError:
    raise SystemExit("huggingface_hub is required: pip install huggingface_hub")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

HF_TOKEN = os.environ.get("HF_TOKEN")
VIEWER_BASE = "https://datasets-server.huggingface.co"

IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
VIDEO_EXTS   = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv", ".m4v", ".flv", ".mpeg", ".mpg"}
PARQUET_EXTS = {".parquet"}
AUDIO_EXTS   = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus"}
ZIP_EXTS     = {".zip", ".tar", ".gz", ".tgz", ".7z", ".rar"}

RATE_LIMIT_SLEEP       = 0.1
MAX_RATE_LIMIT_RETRIES = 1
DEFAULT_RETRY_AFTER    = 1.0
REQ_TIMEOUT            = 5   # Hard timeout of 5 seconds for all network requests

# Global socket timeout to safeguard any non-requests socket calls
socket.setdefaulttimeout(REQ_TIMEOUT)

INPUT_ID_COLUMN_CANDIDATES        = ["Dataset ID", "dataset_id"]
INPUT_NAME_COLUMN_CANDIDATES      = ["Dataset name", "dataset_name"]
INPUT_DESC_COLUMN_CANDIDATES      = ["Description", "description"]
INPUT_DOWNLOADS_COLUMN_CANDIDATES = ["Downloads", "downloads"]
INPUT_LIKES_COLUMN_CANDIDATES     = ["Likes", "likes"]
INPUT_CREATED_COLUMN_CANDIDATES   = ["Created date", "created_at"]

EXTRA_COLUMNS = [
    "file_types_found",
    "image_count",
    "video_count",
    "parquet_count",
    "audio_count",
    "zip_count",
    "other_count",
    "parquet_image_cols",
    "dup_status",
    "duplicate_of",
]


# ─────────────────────────────────────────────────────────────────────────────
# File-type classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_files(files):
    """Bucket a repo's file list by extension."""
    buckets = {"image": [], "video": [], "parquet": [], "audio": [], "zip": [], "other": []}
    for f in files:
        ext = os.path.splitext(f.lower())[1]
        if   ext in IMAGE_EXTS:   buckets["image"].append(f)
        elif ext in VIDEO_EXTS:   buckets["video"].append(f)
        elif ext in PARQUET_EXTS: buckets["parquet"].append(f)
        elif ext in AUDIO_EXTS:   buckets["audio"].append(f)
        elif ext in ZIP_EXTS:     buckets["zip"].append(f)
        else:                     buckets["other"].append(f)
    return buckets


def types_found_str(buckets):
    return ", ".join(t for t in ("image", "video", "parquet") if buckets[t])


def decide_stage1_bucket(buckets):
    """Return 'included', 'needs_review', or 'excluded'."""
    if buckets["image"] or buckets["video"] or buckets["parquet"]:
        return "included"
    if buckets["zip"]:
        return "needs_review"
    return "excluded"


def sub_bucket(has_image, has_video, has_parquet):
    """Split included rows into the 3 final buckets."""
    if has_parquet:
        return "has_parquet"
    if has_image and not has_video:
        return "images_only"
    return "mixed_media"


# ─────────────────────────────────────────────────────────────────────────────
# Parquet schema inspection via HF Dataset Viewer /info (no download)
# ─────────────────────────────────────────────────────────────────────────────

def _viewer_headers():
    h = {"Accept": "application/json"}
    if HF_TOKEN:
        h["Authorization"] = f"Bearer {HF_TOKEN}"
    return h


def _is_image_feature(feat):
    if not isinstance(feat, dict):
        return False
    if feat.get("_type") == "Image":
        return True
    if {"bytes", "path"}.issubset(feat.keys()):
        return True
    return False


def get_parquet_image_cols(dataset_id):
    """Query HF Dataset Viewer /info endpoint with a strict 5s timeout."""
    try:
        resp = requests.get(
            f"{VIEWER_BASE}/info",
            params={"dataset": dataset_id},
            headers=_viewer_headers(),
            timeout=REQ_TIMEOUT,
        )
        if resp.status_code == 401:
            return [], "viewer_unauthorized"
        if resp.status_code == 404:
            return [], "viewer_not_found"
        if resp.status_code != 200:
            return [], f"viewer_http_{resp.status_code}"

        image_cols = set()
        for config_info in resp.json().get("dataset_info", {}).values():
            for col_name, feat in config_info.get("features", {}).items():
                if _is_image_feature(feat):
                    image_cols.add(col_name)

        return sorted(image_cols), None

    except (requests.Timeout, socket.timeout):
        return [], f"timeout_after_{REQ_TIMEOUT}s"
    except Exception as e:
        return [], f"viewer_error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Deduplication
# ─────────────────────────────────────────────────────────────────────────────

def normalize_text(text):
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r"see the full description.*", "", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_name(name):
    if not name:
        return ""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def similarity(a, b):
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def find_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise SystemExit(f"Could not find any of {candidates} in columns: {list(df.columns)}")
    return None


def dedupe(records, desc_sim_threshold, name_sim_threshold):
    n = len(records)
    assigned = [False] * n
    result = {}

    for i in range(n):
        if assigned[i]:
            continue
        group = [i]
        for j in range(i + 1, n):
            if assigned[j]:
                continue
            r_i, r_j = records[i], records[j]
            desc_match = (
                r_i["description_norm"] and r_j["description_norm"]
                and similarity(r_i["description_norm"], r_j["description_norm"]) >= desc_sim_threshold
            )
            name_struct_match = (
                r_i["name_norm"] and r_j["name_norm"]
                and similarity(r_i["name_norm"], r_j["name_norm"]) >= name_sim_threshold
                and r_i["file_types"] == r_j["file_types"]
                and r_i["file_types"] != ""
            )
            if desc_match or name_struct_match:
                group.append(j)

        if len(group) == 1:
            result[records[i]["idx"]] = ("canonical", None)
            assigned[i] = True
            continue

        def rank_key(gi):
            r = records[gi]
            return (-(r["downloads"] + r["likes"]), r["created"] or "9999")

        canonical_local = min(group, key=rank_key)
        canonical_idx = records[canonical_local]["idx"]
        for gi in group:
            assigned[gi] = True
            result[records[gi]["idx"]] = (
                ("canonical", None) if gi == canonical_local
                else ("duplicate", canonical_idx)
            )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# HF direct HTTP fetch with strict 5-second timeout
# ─────────────────────────────────────────────────────────────────────────────

def fetch_repo_files(dataset_id):
    """
    Fetch repository files via direct HTTP request to Hugging Face API
    with a strict REQ_TIMEOUT (5s) timeout. No indefinite blocking.
    """
    url = f"https://huggingface.co/api/datasets/{dataset_id}"
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    try:
        resp = requests.get(url, headers=headers, timeout=REQ_TIMEOUT)
        if resp.status_code == 404:
            return None, "repo_not_found"
        if resp.status_code in (401, 403):
            return None, "gated_or_no_access"
        if resp.status_code == 429:
            return None, "rate_limited"
        if resp.status_code != 200:
            return None, f"http_{resp.status_code}"

        data = resp.json()
        siblings = data.get("siblings") or []
        files = [s.get("rfilename") for s in siblings if s.get("rfilename")]
        return files, None

    except (requests.Timeout, socket.timeout):
        return None, f"timeout_after_{REQ_TIMEOUT}s"
    except Exception as e:
        err_msg = str(e)
        if "timeout" in err_msg.lower():
            return None, f"timeout_after_{REQ_TIMEOUT}s"
        return None, f"error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def open_csv_writer(path, fieldnames, resume):
    """Open a CSV file for writing (or appending on resume). Returns (file, writer)."""
    mode = "a" if resume and os.path.exists(path) else "w"
    f = open(path, mode, newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    if mode == "w":
        writer.writeheader()
    return f, writer


def load_done_ids_from_csvs(id_col_candidates, *paths):
    """Read already-processed dataset IDs from existing CSV files."""
    done = set()
    for path in paths:
        if not os.path.exists(path):
            continue
        try:
            with open(path, newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for col in id_col_candidates:
                        val = (row.get(col) or "").strip()
                        if val:
                            done.add(val)
                            break
        except Exception:
            pass
    return done


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def to_int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input",      required=True, help="Input CSV from fetch_query_metadata_from_huggingface.py")
    parser.add_argument("--output-dir", default=None,  help="Output directory (default: same dir as --input)")
    parser.add_argument("--limit",      type=int, default=None, help="Process only the first N rows (for testing)")
    parser.add_argument("--restart",    action="store_true",
                        help="Ignore existing output CSVs and start fresh from row 1")
    parser.add_argument("--skip-parquet-inspect", action="store_true",
                        help="Skip parquet schema /info check (faster, but no parquet_with_images.csv)")
    parser.add_argument("--desc-sim-threshold", type=float, default=0.90,
                        help="Description similarity threshold for dedup (default 0.90)")
    parser.add_argument("--name-sim-threshold", type=float, default=0.85,
                        help="Name similarity threshold for dedup (default 0.85)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"Input not found: {args.input}")

    df = pd.read_csv(args.input)
    id_col        = find_col(df, INPUT_ID_COLUMN_CANDIDATES)
    name_col      = find_col(df, INPUT_NAME_COLUMN_CANDIDATES,      required=False)
    desc_col      = find_col(df, INPUT_DESC_COLUMN_CANDIDATES,      required=False)
    downloads_col = find_col(df, INPUT_DOWNLOADS_COLUMN_CANDIDATES, required=False)
    likes_col     = find_col(df, INPUT_LIKES_COLUMN_CANDIDATES,     required=False)
    created_col   = find_col(df, INPUT_CREATED_COLUMN_CANDIDATES,   required=False)

    if args.limit:
        df = df.head(args.limit)

    base_name = os.path.splitext(os.path.basename(args.input))[0]
    out_dir   = args.output_dir or os.path.dirname(os.path.abspath(args.input))
    os.makedirs(out_dir, exist_ok=True)

    fieldnames = list(df.columns) + EXTRA_COLUMNS

    # Stage 1 intermediate raw included file (directly appended per row)
    raw_included_path = os.path.join(out_dir, f".{base_name}_raw_included.csv")

    paths = {
        "needs_review":        os.path.join(out_dir, f"filtered_{base_name}_needs_review.csv"),
        "excluded":            os.path.join(out_dir, f"filtered_{base_name}_excluded.csv"),
        "main":                os.path.join(out_dir, f"filtered_{base_name}_main.csv"),
        "images_only":         os.path.join(out_dir, f"filtered_{base_name}_images_only.csv"),
        "has_parquet":         os.path.join(out_dir, f"filtered_{base_name}_has_parquet.csv"),
        "parquet_with_images": os.path.join(out_dir, f"filtered_{base_name}_parquet_with_images.csv"),
        "mixed_media":         os.path.join(out_dir, f"filtered_{base_name}_mixed_media.csv"),
        "duplicates":          os.path.join(out_dir, f"filtered_{base_name}_duplicates.csv"),
    }

    if args.restart:
        for p in [raw_included_path] + list(paths.values()):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        done_ids = set()
        resuming = False
    else:
        done_ids = load_done_ids_from_csvs(
            INPUT_ID_COLUMN_CANDIDATES,
            raw_included_path, paths["needs_review"], paths["excluded"]
        )
        resuming = bool(done_ids)

    total = len(df)
    skip_count = sum(1 for _, row in df.iterrows() if str(row[id_col]).strip() in done_ids)

    print(f"Fetching/classifying {total} dataset(s) from {args.input} ...")
    if resuming:
        print(f"Resuming directly from CSVs: {skip_count} already recorded, {total - skip_count} remaining.")
    if not HF_TOKEN:
        print("NOTE: no HF_TOKEN set - unauthenticated requests get a lower rate limit.")

    # ── Stage 1: stream rows directly into CSVs as they're fetched ───────────
    fetch_errors = Counter()
    stage1_counts = {"included": 0, "needs_review": 0, "excluded": 0}

    f_inc, w_inc = open_csv_writer(raw_included_path,    fieldnames, resuming)
    f_nr,  w_nr  = open_csv_writer(paths["needs_review"], fieldnames, resuming)
    f_exc, w_exc = open_csv_writer(paths["excluded"],     fieldnames, resuming)

    try:
        for i, row in df.iterrows():
            dataset_id = str(row[id_col]).strip()

            if dataset_id in done_ids:
                continue

            out_row = row.to_dict()
            print(f"[{i+1}/{total}] {dataset_id} ...", end=" ", flush=True)

            files, err = fetch_repo_files(dataset_id)
            time.sleep(RATE_LIMIT_SLEEP)

            if err or files is None:
                fetch_errors[err or "unknown_error"] += 1
                out_row.update({
                    "file_types_found": "", "image_count": 0, "video_count": 0,
                    "parquet_count": 0, "audio_count": 0, "zip_count": 0,
                    "other_count": 0, "parquet_image_cols": "",
                    "dup_status": "", "duplicate_of": "",
                })
                w_exc.writerow(out_row)
                f_exc.flush()
                stage1_counts["excluded"] += 1
                done_ids.add(dataset_id)
                print(f"FETCH FAILED ({err}) -> excluded")
                continue

            buckets = classify_files(files)
            bucket  = decide_stage1_bucket(buckets)
            out_row.update({
                "file_types_found":   types_found_str(buckets),
                "image_count":        len(buckets["image"]),
                "video_count":        len(buckets["video"]),
                "parquet_count":      len(buckets["parquet"]),
                "audio_count":        len(buckets["audio"]),
                "zip_count":          len(buckets["zip"]),
                "other_count":        len(buckets["other"]),
                "parquet_image_cols": "",
                "dup_status":         "",
                "duplicate_of":       "",
            })

            if bucket == "included":
                w_inc.writerow(out_row)
                f_inc.flush()
            elif bucket == "needs_review":
                w_nr.writerow(out_row)
                f_nr.flush()
            else:
                w_exc.writerow(out_row)
                f_exc.flush()

            stage1_counts[bucket] += 1
            done_ids.add(dataset_id)
            print(bucket)

    except KeyboardInterrupt:
        print("\nInterrupted - all progress was written directly to CSVs. Re-run to resume.")
    finally:
        f_inc.close()
        f_nr.close()
        f_exc.close()

    # ── Stage 2: read raw_included CSV → deduplicate ─────────────────────────
    if not os.path.exists(raw_included_path):
        print("No included datasets found. Done.")
        return

    with open(raw_included_path, newline="", encoding="utf-8-sig") as f:
        included = list(csv.DictReader(f))

    if not included:
        print("No included datasets recorded. Done.")
        return

    for r in included:
        for col in ("image_count", "video_count", "parquet_count",
                    "audio_count", "zip_count", "other_count"):
            r[col] = to_int(r.get(col, 0))

    records = []
    for local_idx, r in enumerate(included):
        desc = r.get(desc_col, "") if desc_col else ""
        name = r.get(name_col, "") if name_col else r.get(id_col, "")
        records.append({
            "idx":              local_idx,
            "description_norm": normalize_text(desc),
            "name_norm":        normalize_name(name),
            "file_types":       r.get("file_types_found", ""),
            "downloads":        to_int(r.get(downloads_col, 0)) if downloads_col else 0,
            "likes":            to_int(r.get(likes_col, 0))     if likes_col     else 0,
            "created":          str(r.get(created_col, ""))     if created_col   else "",
        })

    dup_result = dedupe(records, args.desc_sim_threshold, args.name_sim_threshold)

    main_rows, duplicate_rows = [], []
    id_by_local_idx = {idx: included[idx][id_col] for idx in range(len(included))}
    for local_idx, r in enumerate(included):
        status, canonical_local_idx = dup_result[local_idx]
        if status == "canonical":
            r["dup_status"] = "canonical"
            main_rows.append(r)
        else:
            r["dup_status"] = "duplicate"
            r["duplicate_of"] = id_by_local_idx[canonical_local_idx]
            duplicate_rows.append(r)

    # ── Stage 3: split main into images_only / has_parquet / mixed_media ─────
    images_only, has_parquet_rows, mixed_media = [], [], []
    for r in main_rows:
        sub = sub_bucket(r["image_count"] > 0, r["video_count"] > 0, r["parquet_count"] > 0)
        if   sub == "images_only": images_only.append(r)
        elif sub == "has_parquet": has_parquet_rows.append(r)
        else:                      mixed_media.append(r)

    # ── Stage 4: parquet schema inspection (streaming to CSV immediately) ────
    parquet_inspect_errors = Counter()
    parquet_with_images_count = 0

    f_piw, w_piw = open_csv_writer(paths["parquet_with_images"], fieldnames, resume=False)
    try:
        if not args.skip_parquet_inspect and has_parquet_rows:
            n_parquet = len(has_parquet_rows)
            print(f"\nInspecting parquet schemas for {n_parquet} dataset(s) via HF Dataset Viewer /info (5s timeout) ...")
            for idx, r in enumerate(has_parquet_rows):
                dataset_id = str(r[id_col]).strip()
                print(f"  [{idx+1}/{n_parquet}] {dataset_id} ...", end=" ", flush=True)

                image_cols, err = get_parquet_image_cols(dataset_id)

                if err:
                    parquet_inspect_errors[err] += 1
                    r["parquet_image_cols"] = ""
                    print(f"skipped ({err})")
                elif image_cols:
                    r["parquet_image_cols"] = ", ".join(image_cols)
                    w_piw.writerow(r)
                    f_piw.flush()
                    parquet_with_images_count += 1
                    print(f"image cols found: {image_cols}")
                else:
                    r["parquet_image_cols"] = ""
                    print("no image cols")

                time.sleep(RATE_LIMIT_SLEEP)
    finally:
        f_piw.close()

    # ── Stage 5: write remaining output CSVs ─────────────────────────────────
    write_csv(paths["main"],       fieldnames, main_rows)
    write_csv(paths["images_only"],fieldnames, images_only)
    write_csv(paths["has_parquet"],fieldnames, has_parquet_rows)
    write_csv(paths["mixed_media"],fieldnames, mixed_media)
    write_csv(paths["duplicates"], fieldnames, duplicate_rows)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"  Included (pre-dedup):      {len(included)}")
    print(f"  main.csv (deduped):        {len(main_rows)}  ({len(duplicate_rows)} duplicates removed)")
    print(f"    images_only:             {len(images_only)}")
    print(f"    has_parquet:             {len(has_parquet_rows)}")
    if not args.skip_parquet_inspect:
        print(f"    parquet_with_images:     {parquet_with_images_count}")
        if parquet_inspect_errors:
            print(f"    parquet inspect errors:  {dict(parquet_inspect_errors)}")
    print(f"    mixed_media:             {len(mixed_media)}")
    print(f"  needs_review:              {stage1_counts['needs_review']}")
    print(f"  excluded:                  {stage1_counts['excluded']}")
    if fetch_errors:
        print(f"  file-listing errors:       {dict(fetch_errors)}")
    print("\nWritten:")
    for name, path in paths.items():
        if name == "parquet_with_images" and args.skip_parquet_inspect:
            continue
        print(f"  {name:24s} -> {path}")
    print(f"\n  (Resume tracking file: {raw_included_path})")
    print("  Re-run the same command to resume if interrupted.")
    print("  Use --restart to discard previous results and start fresh.")


if __name__ == "__main__":
    main()