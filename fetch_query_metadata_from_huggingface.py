#fetch_query_metadata_from_huggingface.py
 
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm


# ============================================================
# DEFAULT CONFIGURATION (Overridden by CLI arguments)
# ============================================================

DEFAULT_SEARCH_QUERY = ""
DEFAULT_SORT = "trendingScore"

# Set to None to retrieve ALL matching datasets.
DEFAULT_LIMIT = None

# Backwards-compatibility aliases
SEARCH_QUERY = DEFAULT_SEARCH_QUERY
SORT = DEFAULT_SORT
LIMIT = DEFAULT_LIMIT

# Viewer API requests can be numerous. Pause between requests.
REQUEST_DELAY = 0.50

HF_DATASETS_API = "https://huggingface.co/api/datasets"
PAGE_SIZE = 30

# Timeout for Dataset Viewer HTTP requests.
HTTP_TIMEOUT = 30

# Optional:
# Set your HF token as an environment variable:
#
#   Linux/macOS:
#       export HF_TOKEN="hf_xxxxxxxxx"
#
#   Windows PowerShell:
#       $env:HF_TOKEN="hf_xxxxxxxxx"
#
# Public dataset metadata/search normally works without a token,
# but a token can help with gated/private metadata.
import os

HF_TOKEN = os.getenv("HF_TOKEN")


# ============================================================
# HELPERS
# ============================================================

VIEWER_API = "https://datasets-server.huggingface.co"

session = requests.Session()

if HF_TOKEN:
    session.headers.update({
        "Authorization": f"Bearer {HF_TOKEN}"
    })


def safe_json(value: Any) -> str:
    """
    Convert nested HF metadata into a JSON string so it fits
    cleanly inside one Excel cell.
    """
    if value is None:
        return ""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            default=str
        )
    except Exception:
        return str(value)


def safe_get(obj: Any, key: str, default=None):
    """
    Safely retrieve fields from API dictionaries or HF objects.
    """

    cli_keys = {
        "cardData": "card_data",
        "createdAt": "created_at",
        "lastModified": "last_modified",
        "mainSize": "main_size",
        "trendingScore": "trending_score",
    }

    if isinstance(obj, dict):
        return obj.get(
            key,
            obj.get(cli_keys.get(key, key), default)
        )

    try:
        return getattr(obj, key, default)
    except Exception:
        return default


def get_output_paths(search_query: str, custom_output_dir: str | Path | None = None):
    """
    Generate dynamic output paths based on the search query.
    E.g. query 'drone' -> folder 'hf_drone_candidates/'
         query 'uav'   -> folder 'hf_uav_candidates/'
    """
    safe_slug = "".join(
        c if c.isalnum() or c in ("-", "_") else "_" for c in search_query
    ).strip("_").lower()

    if not safe_slug:
        safe_slug = "dataset"

    if custom_output_dir:
        output_dir = Path(custom_output_dir)
    else:
        output_dir = Path(f"hf_{safe_slug}_candidates")

    output_xlsx = output_dir / f"hf_{safe_slug}_candidates.xlsx"
    output_csv = output_dir / f"hf_{safe_slug}_candidates.csv"
    output_raw_json = output_dir / f"hf_{safe_slug}_raw.json"

    return output_dir, output_xlsx, output_csv, output_raw_json


def get_hf_datasets(
    search_query: str = DEFAULT_SEARCH_QUERY,
    sort: str = DEFAULT_SORT,
    limit: int | None = DEFAULT_LIMIT,
):
    """
    Retrieve all datasets matching the search query with pagination.
    """

    all_datasets = []
    seen_ids = set()
    next_url = HF_DATASETS_API
    params = {
        "search": search_query,
        "sort": sort,
        "limit": PAGE_SIZE,
    }

    headers = {}

    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    while next_url:
        print(f"Fetching datasets limit={PAGE_SIZE}")

        response = requests.get(
            next_url,
            params=params,
            headers=headers,
            timeout=60,
        )

        print(
            f"HTTP {response.status_code} {response.url}"
        )

        if response.status_code != 200:
            print("HF API response:")
            print(response.text)
            response.raise_for_status()

        batch = response.json()

        if not batch:
            print("No more datasets.")
            break

        new_datasets = [
            dataset for dataset in batch
            if dataset.get("id") not in seen_ids
        ]

        seen_ids.update(
            dataset.get("id") for dataset in new_datasets
        )
        all_datasets.extend(new_datasets)

        print(
            f"Received: {len(batch)} "
            f"| New: {len(new_datasets)} | "
            f"Total: {len(all_datasets)}"
        )

        if limit is not None and len(all_datasets) >= limit:
            all_datasets = all_datasets[:limit]
            break

        next_url = response.links.get("next", {}).get("url")
        params = None

        if not next_url:
            print("No more datasets.")
            break

        time.sleep(REQUEST_DELAY)

    return all_datasets


def request_json(url: str, params: dict):
    """
    GET JSON from an HF Dataset Viewer endpoint.

    Returns:
        dict or None
    """
    try:
        response = session.get(
            url,
            params=params,
            timeout=HTTP_TIMEOUT
        )

        if response.status_code == 200:
            return response.json()

        print(
            f"Viewer request failed: "
            f"{response.status_code} "
            f"{response.url}"
        )

    except requests.RequestException as exc:
        print(f"Viewer request error: {exc}")

    return None


# ============================================================
# DATASET VIEWER ENRICHMENT
# ============================================================

def get_viewer_info(dataset_id: str) -> dict:
    """
    Query the Hugging Face Dataset Viewer for:
      - viewer validity
      - configurations
      - splits
      - number of rows
      - size information
    """

    result = {
        "Dataset Viewer available?": "",
        "Viewer preview available?": "",
        "Viewer search available?": "",
        "Viewer statistics available?": "",
        "Configurations": "",
        "Splits": "",
        "Number of candidate rows": "",
        "Dataset Viewer size bytes": "",
        "Dataset Viewer original bytes": "",
        "Dataset Viewer parquet bytes": "",
        "Viewer pending": "",
        "Viewer failed": "",
    }

    # --------------------------------------------------------
    # /is-valid
    # --------------------------------------------------------

    valid = request_json(
        f"{VIEWER_API}/is-valid",
        {"dataset": dataset_id}
    )

    if valid:
        result["Dataset Viewer available?"] = valid.get(
            "viewer", ""
        )

        result["Viewer preview available?"] = valid.get(
            "preview", ""
        )

        result["Viewer search available?"] = valid.get(
            "search", ""
        )

        result["Viewer statistics available?"] = valid.get(
            "statistics", ""
        )

    time.sleep(REQUEST_DELAY)

    # --------------------------------------------------------
    # /splits
    # --------------------------------------------------------

    splits_response = request_json(
        f"{VIEWER_API}/splits",
        {"dataset": dataset_id}
    )

    if splits_response:
        splits = splits_response.get("splits", [])

        configs = sorted(
            {
                s.get("config")
                for s in splits
                if s.get("config") is not None
            }
        )

        split_names = sorted(
            {
                s.get("split")
                for s in splits
                if s.get("split") is not None
            }
        )

        result["Configurations"] = safe_json(configs)
        result["Splits"] = safe_json(split_names)

        result["Viewer pending"] = safe_json(
            splits_response.get("pending", [])
        )

        result["Viewer failed"] = safe_json(
            splits_response.get("failed", [])
        )

    time.sleep(REQUEST_DELAY)

    # --------------------------------------------------------
    # /size
    # --------------------------------------------------------

    size_response = request_json(
        f"{VIEWER_API}/size",
        {"dataset": dataset_id}
    )

    if size_response:
        size = size_response.get("size", {})
        dataset_size = size.get("dataset", {})

        result["Number of candidate rows"] = dataset_size.get(
            "num_rows",
            ""
        )

        result["Dataset Viewer original bytes"] = dataset_size.get(
            "num_bytes_original_files",
            ""
        )

        result["Dataset Viewer parquet bytes"] = dataset_size.get(
            "num_bytes_parquet_files",
            ""
        )

        result["Dataset Viewer size bytes"] = dataset_size.get(
            "num_bytes_memory",
            ""
        )

    return result


# ============================================================
# MAIN EXTRACTION
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Retrieve Hugging Face dataset candidates for a given search query."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=DEFAULT_SEARCH_QUERY,
        help=f"Search query term for Hugging Face datasets (default: '{DEFAULT_SEARCH_QUERY}')",
    )
    parser.add_argument(
        "--query", "-q",
        dest="query_flag",
        default=None,
        help="Search query term (overrides positional argument if provided)",
    )
    parser.add_argument(
        "--sort", "-s",
        default=DEFAULT_SORT,
        help=f"Sort criteria for dataset search (default: '{DEFAULT_SORT}')",
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=DEFAULT_LIMIT,
        help="Maximum number of datasets to retrieve (default: retrieve all)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory path (default: hf_<query>_candidates)",
    )

    args = parser.parse_args()

    search_query = args.query_flag if args.query_flag is not None else args.query
    sort = args.sort
    limit = args.limit

    output_dir, output_xlsx, output_csv, output_raw_json = get_output_paths(
        search_query=search_query,
        custom_output_dir=args.output_dir,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    retrieval_timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    print("=" * 70)
    print("HUGGING FACE DATASET CANDIDATE EXTRACTION")
    print("=" * 70)

    print(f"Search query       : {search_query}")
    print(f"Sort               : {sort}")
    print(f"Limit              : {limit}")
    print(f"Retrieved at (UTC) : {retrieval_timestamp}")
    print()

    # --------------------------------------------------------
    # IMPORTANT
    #
    # The CLI returns the matching dataset records as JSON.
    # --------------------------------------------------------

    datasets = get_hf_datasets(
        search_query=search_query,
        sort=sort,
        limit=limit,
    )

    rows = []
    raw_records = []

    print("Retrieving matching repositories...")

    for rank, dataset in enumerate(
        tqdm(datasets, desc="HF datasets"),
        start=1
    ):

        dataset_id = safe_get(
            dataset,
            "id",
            ""
        )

        if not dataset_id:
            continue

        # ----------------------------------------------------
        # Basic HF metadata
        # ----------------------------------------------------

        author = safe_get(
            dataset,
            "author",
            ""
        )

        description = safe_get(
            dataset,
            "description",
            ""
        )

        tags = safe_get(
            dataset,
            "tags",
            []
        )

        card_data = safe_get(
            dataset,
            "cardData",
            None
        )

        citation = safe_get(
            dataset,
            "citation",
            ""
        )

        siblings = safe_get(
            dataset,
            "siblings",
            []
        )

        created_at = safe_get(
            dataset,
            "createdAt",
            ""
        )

        last_modified = safe_get(
            dataset,
            "lastModified",
            ""
        )

        downloads = safe_get(
            dataset,
            "downloads",
            ""
        )

        likes = safe_get(
            dataset,
            "likes",
            ""
        )

        trending_score = safe_get(
            dataset,
            "trendingScore",
            ""
        )

        main_size = safe_get(
            dataset,
            "mainSize",
            ""
        )

        sha = safe_get(
            dataset,
            "sha",
            ""
        )

        paperswithcode_id = safe_get(
            dataset,
            "paperswithcode_id",
            ""
        )

        # ----------------------------------------------------
        # Dataset name
        # ----------------------------------------------------

        dataset_name = dataset_id.split(
            "/"
        )[-1]

        # ----------------------------------------------------
        # Repository files
        # ----------------------------------------------------

        repository_files = []

        for sibling in siblings or []:

            if isinstance(
                sibling,
                dict
            ):
                filename = sibling.get(
                    "rfilename",
                    ""
                )
            else:
                filename = safe_get(
                    sibling,
                    "rfilename",
                    ""
                )

            if filename:
                repository_files.append(
                    filename
                )

        # ----------------------------------------------------
        # HF URL
        # ----------------------------------------------------

        hf_url = (
            f"https://huggingface.co/datasets/"
            f"{dataset_id}"
        )

        # ----------------------------------------------------
        # Dataset card
        # ----------------------------------------------------

        # cardData is structured metadata from the Hub.
        # Keep the entire object rather than throwing information
        # away at this stage.
        card_data_json = safe_json(
            card_data
        )

        # Extract common license representations when possible.
        license_value = ""

        if isinstance(card_data, dict):
            license_value = (
                card_data.get("license")
                or card_data.get("license_name")
                or ""
            )

        # ----------------------------------------------------
        # Viewer enrichment
        # ----------------------------------------------------

        viewer_info = {
            "Dataset Viewer available?": "",
            "Viewer preview available?": "",
            "Viewer search available?": "",
            "Viewer statistics available?": "",
            "Configurations": "",
            "Splits": "",
            "Number of candidate rows": "",
            "Dataset Viewer size bytes": "",
            "Dataset Viewer original bytes": "",
            "Dataset Viewer parquet bytes": "",
            "Viewer pending": "",
            "Viewer failed": "",
        }

        # ----------------------------------------------------
        # Final spreadsheet row
        # ----------------------------------------------------

        row = {

            # -----------------------------------------------
            # Identity
            # -----------------------------------------------

            "Dataset ID": dataset_id,

            "Dataset name": dataset_name,

            "Author / organization": author,

            "URL": hf_url,

            # -----------------------------------------------
            # Description / metadata
            # -----------------------------------------------

            "Description": description,

            "Tags": safe_json(tags),

            "License": license_value,

            "HF cardData": card_data_json,

            "Citation": citation,

            # -----------------------------------------------
            # Dates
            # -----------------------------------------------

            "Created date": created_at,

            "Last modified": last_modified,

            # -----------------------------------------------
            # Popularity / ranking
            # -----------------------------------------------

            "Downloads": downloads,

            "Likes": likes,

            "Trending score": trending_score,

            "Search rank": rank,

            # -----------------------------------------------
            # Storage
            # -----------------------------------------------

            "Dataset size": main_size,

            # -----------------------------------------------
            # Reproducibility
            # -----------------------------------------------

            "SHA / revision": sha,

            # -----------------------------------------------
            # External identifiers
            # -----------------------------------------------

            "PapersWithCode ID": paperswithcode_id,

            # -----------------------------------------------
            # Repository
            # -----------------------------------------------

            "Repository files": safe_json(
                repository_files
            ),

            "Repository file count": len(
                repository_files
            ),

            # -----------------------------------------------
            # Dataset Viewer
            # -----------------------------------------------

            **viewer_info,

            # -----------------------------------------------
            # Search provenance
            # -----------------------------------------------

            "Search query": search_query,

            "Search sort": sort,

            "Retrieval timestamp UTC":
                retrieval_timestamp,

            "HF URL": hf_url,
        }

        rows.append(row)

        # Preserve raw HF object separately.
        try:
            raw_records.append(dataset)
        except Exception:
            raw_records.append(
                {
                    "id": dataset_id,
                    "raw_repr": repr(dataset)
                }
            )

    # ========================================================
    # RESULT COUNT
    # ========================================================

    count = len(rows)

    print()
    print("=" * 70)
    print(f"TOTAL MATCHING DATASETS: {count}")
    print("=" * 70)

    # This is deliberately NOT an assertion against 401.
    #
    # If HF returns 401 -> output contains 401.
    # If HF returns 145 -> output contains 145.
    # If HF returns 600 -> output contains 600.

    # ========================================================
    # DATAFRAME
    # ========================================================

    df = pd.DataFrame(rows)

    # Stable ordering.
    preferred_columns = [
        "Search rank",
        "Dataset ID",
        "Dataset name",
        "Author / organization",
        "URL",
        "HF URL",
        "Description",
        "Tags",
        "License",
        "HF cardData",
        "Created date",
        "Last modified",
        "Downloads",
        "Likes",
        "Trending score",
        "Dataset size",
        "SHA / revision",
        "PapersWithCode ID",
        "Citation",
        "Repository files",
        "Repository file count",
        "Dataset Viewer available?",
        "Viewer preview available?",
        "Viewer search available?",
        "Viewer statistics available?",
        "Configurations",
        "Splits",
        "Number of candidate rows",
        "Dataset Viewer size bytes",
        "Dataset Viewer original bytes",
        "Dataset Viewer parquet bytes",
        "Viewer pending",
        "Viewer failed",
        "Search query",
        "Search sort",
        "Retrieval timestamp UTC",
    ]

    existing_columns = [
        c for c in preferred_columns
        if c in df.columns
    ]

    remaining_columns = [
        c for c in df.columns
        if c not in existing_columns
    ]

    df = df[
        existing_columns + remaining_columns
    ]

    # ========================================================
    # SAVE CSV
    # ========================================================

    df.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # SAVE EXCEL
    # ========================================================

    with pd.ExcelWriter(
        output_xlsx,
        engine="openpyxl"
    ) as writer:

        # Main candidate universe.
        df.to_excel(
            writer,
            sheet_name="Candidates",
            index=False
        )

        # Extraction metadata.
        metadata = pd.DataFrame([
            {
                "Parameter": "Search query",
                "Value": search_query
            },
            {
                "Parameter": "Sort",
                "Value": sort
            },
            {
                "Parameter": "Limit",
                "Value": str(limit)
            },
            {
                "Parameter": "Total datasets retrieved",
                "Value": count
            },
            {
                "Parameter": "Retrieval timestamp UTC",
                "Value": retrieval_timestamp
            },
            {
                "Parameter": "Hub API",
                "Value": "Hugging Face Hub API"
            },
            {
                "Parameter": "Dataset Viewer API",
                "Value": VIEWER_API
            },
        ])

        metadata.to_excel(
            writer,
            sheet_name="Extraction Metadata",
            index=False
        )

        # Simple repository-file inventory.
        file_rows = []

        for row in rows:

            dataset_id = row[
                "Dataset ID"
            ]

            files = json.loads(
                row["Repository files"]
            )

            for filename in files:

                file_rows.append({
                    "Dataset ID": dataset_id,
                    "Repository file": filename
                })

        files_df = pd.DataFrame(
            file_rows
        )

        if not files_df.empty:
            files_df.to_excel(
                writer,
                sheet_name="Repository Files",
                index=False
            )

    # ========================================================
    # SAVE RAW JSON
    # ========================================================

    with open(
        output_raw_json,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            raw_records,
            f,
            ensure_ascii=False,
            indent=2,
            default=str
        )

    print()
    print("Files written:")
    print(f"  Excel : {output_xlsx}")
    print(f"  CSV   : {output_csv}")
    print(f"  Raw   : {output_raw_json}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()