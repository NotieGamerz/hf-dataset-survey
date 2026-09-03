#!/usr/bin/env python3
"""
validate_datasets.py

Classify drone dataset samples using local VLM server(s) (OpenAI-compatible
API), and append aggregated results to a resumable output CSV.

v7: split image and video inference across two independently-configurable
backends. Images go to whatever local server you already have running
(e.g. mlx-omni-server / OMLX on Mac, default http://127.0.0.1:8000/v1) using
base64-encoded payloads, since that's what those servers actually accept.
Videos go to a vLLM server (which supports native video_url input and, on
the same machine, cheap file:// local-path media) - this is NOT required
unless you actually have a video to classify, so nothing forces a vLLM
install just to run images.

If no separate video backend is configured, video calls fall back to the
same server/model used for images (useful if you're running vLLM for
everything on Linux).

Media transport:
  - images: base64 data URI by default (works against mlx-omni-server,
    ollama, vLLM, etc. without extra server flags)
  - videos: local file:// path by default (cheap, requires vLLM's
    --allowed-local-media-path to cover --images-dir) - falls back to
    base64 if you set --video-media-mode base64

Usage:
    python classify_drone_datasets.py --input input.csv --output output.csv \
        --n-images 100

    # videos routed to a separate vLLM server on Linux, images stay on
    # the local mlx-omni-server:
    python classify_drone_datasets.py --input input.csv --output output.csv \
        --video-vlm-base-url http://192.168.1.50:8000/v1 \
        --video-vlm-model Qwen/Qwen2.5-VL-7B-Instruct \
        --video-vlm-api-key EMPTY

Input CSV needs at least a `dataset_id` column. Optional columns `License`/
`license`, `Tags`/`tags`, `Last modified`/`last_modified`/`Created date` are
used (if present) to fill licensing/last-updated deterministically.

Things worth checking on your vLLM setup before a full run (video only):
  - Server started with enough --limit-mm-per-prompt for video (some vLLM
    versions default to video=0 unless explicitly enabled).
  - --max-model-len large enough for a 30s clip's worth of visual tokens -
    if you get context-length errors, either raise --max-model-len on the
    vLLM server or lower --video-window-seconds here.
  - If using local-path mode for video, --allowed-local-media-path on the
    vLLM server must cover the absolute path of --images-dir, or requests
    will 400.

Requires:
    pip install --break-system-packages requests huggingface_hub pillow
    ffmpeg + ffprobe must be on PATH (still used for trimming clips and
    reading video duration/dimensions - NOT for frame extraction anymore).
"""

import argparse
import base64
import csv
import json
import os
import random
import re
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
# pyrefly: ignore [missing-import]
from PIL import Image
def _downscale_for_vlm(path, max_dim=1024):
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((max_dim, max_dim))
        im.save(path, "JPEG", quality=85)
try:
# pyrefly: ignore [missing-import]
    from huggingface_hub import HfApi
except ImportError:
    HfApi = None

# ------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------
# Default ("image") backend - whatever local server you already run.
VLM_BASE_URL = os.environ.get("VLM_BASE_URL", "http://192.168.0.190:1234")
VLM_MODEL = os.environ.get("VLM_MODEL") or "qwen/qwen3-vl-8b"
VLM_API_KEY = os.environ.get("VLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "admin"

# Optional separate video backend (e.g. vLLM on a Linux box). If unset,
# video calls fall back to the image backend above.
VIDEO_VLM_BASE_URL = os.environ.get("VIDEO_VLM_BASE_URL") or None
VIDEO_VLM_MODEL = os.environ.get("VIDEO_VLM_MODEL") or None
VIDEO_VLM_API_KEY = os.environ.get("VIDEO_VLM_API_KEY") or None

REQUEST_TIMEOUT = 120
DOWNLOAD_TIMEOUT = 30
API_TIMEOUT = 15
MAX_RETRIES = 3
RETRY_SLEEP_SEC = 3
DEFAULT_N_IMAGES = 100
DEFAULT_VIDEO_START_FRACTION = 0.20
DEFAULT_VIDEO_WINDOW_SECONDS = 30
DEFAULT_MAX_IMAGE_MB = 50
DEFAULT_MAX_VIDEO_MB = 200
RAW_POOL_CAP = 500

HF_TOKEN = os.environ.get("HF_TOKEN")
AUTH_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

DATASETS_SERVER_BASE = "https://datasets-server.huggingface.co"

NON_CONTENT_PATTERNS = ("banner", "logo", "thumbnail", "icon", "readme")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v")

DOWNLOAD_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://huggingface.co/",
    "Origin": "https://huggingface.co",
}

SESSION = requests.Session()
SESSION.headers.update(DOWNLOAD_HEADERS)

OUTPUT_COLUMNS = [
    "dataset_id",
    "Image / Video",
    "From drone / Of drone",
    "From drone / Of drone - count",
    "From drone / Of drone - total",
    "Within dataset - do you have more closeups or more distant shots (of drones)",
    "Within dataset - do you have more closeups or more distant shots (of drones) - count",
    "Within dataset - do you have more closeups or more distant shots (of drones) - total",
    "Does dataset also have non-drone flying objects (planes/birds)",
    "Does dataset also have non-drone flying objects (planes/birds) - count",
    "Does dataset also have non-drone flying objects (planes/birds) - total",
    "Day or night",
    "Day or night - count",
    "Day or night - total",
    "Visual or Infra red images",
    "Visual or Infra red images - count",
    "Visual or Infra red images - total",
    "Type of drone (fixed wing / rotary)",
    "Type of drone (fixed wing / rotary) - count",
    "Type of drone (fixed wing / rotary) - total",
    "Resolution of images (majority)",
    "Resolution of images (majority) - count",
    "Resolution of images (majority) - total",
    "Availability and licensing (free/which license)",
    "How recently updated?",
    "Used as training set for which model?",
    "--> Are the model weights available",
]

DETERMINISTIC_NA_FIELDS = [
    "Used as training set for which model?",
    "--> Are the model weights available",
]

IMAGE_SYSTEM_PROMPT = """You are an expert image analyst helping classify aerial / drone-related dataset sample images for a research survey.

You will be shown ONE sample still image from a Hugging Face dataset whose name suggests it relates to drones/UAVs. Look carefully at the image and answer the following fields.

Respond with ONLY a single valid JSON object (no markdown fences, no commentary, no preamble), with exactly these keys:

{
  "image_description": "<one sentence, concrete visual description of what's in the image>",
  "from_drone_or_of_drone": "<Decide using this priority order: (1) If one or more drones/UAVs are visible AS SUBJECTS in the frame - regardless of camera altitude, angle, or how far away they are - answer 'Of drone'. (2) Only if NO drone is visible anywhere in the frame, and the image is a scene/landscape/ground shot taken from an aerial or elevated viewpoint (i.e. what a drone's camera would see looking down/out, with no drone itself pictured), answer 'From drone'. (3) If you cannot tell either way, answer 'Unclear'. A photo of a drone flying over a landscape - even a distant or small drone in the corner of the frame - is 'Of drone', not 'From drone', because the drone itself is what's depicted.>",
  "closeup_or_distant": "<'closeup' if the main subject (drone or ground scene) fills much of the frame / is near, 'distant' if the subject is small/far away or the shot is a wide aerial view, or 'NA' if not applicable>",
  "non_drone_flying_objects": "<'Yes' if you can see planes, birds, helicopters or other non-drone flying objects in the image, 'No' if you don't see any, or 'Not specified / NA' if you cannot judge from a single image>",
  "day_or_night": "<'Day', 'Night', 'Dusk/Dawn', or 'Unclear'>",
  "visual_or_infrared": "<'Visual' for a normal RGB/color or grayscale photographic image, 'Infrared' if it looks like a thermal/IR image (false-color heat map, monochrome with heat signatures, etc.), or 'Unclear'>",
  "drone_type": "<'Fixed wing' if a fixed-wing drone/aircraft is visible, 'Rotary' if a multirotor/quadcopter/helicopter-style drone is visible, or 'NA' if no drone is visible in the image or type cannot be determined>"
}

Rules:
- Base every answer ONLY on what is visibly present in the image. Do not guess about things outside the image (like dataset licensing, resolution, or which model it trains) - those are not being asked here.
- If something is genuinely not visible or not determinable from a single still image, use "NA", "Unclear", or "Not specified / NA" as appropriate rather than inventing an answer.
- Output must be strictly valid JSON, with double-quoted keys and string values, and nothing else in your response.
"""

VIDEO_SYSTEM_PROMPT = """You are an expert video analyst helping classify aerial / drone-related dataset sample video clips for a research survey.

You will be shown a short VIDEO CLIP from a Hugging Face dataset whose name suggests it relates to drones/UAVs. Watch it carefully and answer the following fields, considering the clip AS A WHOLE (not just its first frame) - if something varies across the clip (e.g. day in some parts, or the shot moves from distant to closeup), pick whichever best characterizes the majority of the clip, or 'Unclear' if it's genuinely mixed/ambiguous.

Respond with ONLY a single valid JSON object (no markdown fences, no commentary, no preamble), with exactly these keys:

{
  "image_description": "<one sentence, concrete description of what's happening in the clip>",
  "from_drone_or_of_drone": "<Decide using this priority order: (1) If one or more drones/UAVs are visible AS SUBJECTS anywhere in the clip - regardless of camera altitude, angle, or how far away they are - answer 'Of drone'. (2) Only if NO drone is visible anywhere in the clip, and it's footage taken from an aerial/elevated viewpoint (i.e. what a drone's camera would see looking down/out, with no drone itself pictured), answer 'From drone'. (3) If you cannot tell either way, answer 'Unclear'.>",
  "closeup_or_distant": "<'closeup' if the main subject fills much of the frame for most of the clip, 'distant' if it's small/far away or a wide aerial view for most of the clip, or 'NA' if not applicable>",
  "non_drone_flying_objects": "<'Yes' if you see planes, birds, helicopters or other non-drone flying objects anywhere in the clip, 'No' if you don't see any, or 'Not specified / NA' if you cannot judge>",
  "day_or_night": "<'Day', 'Night', 'Dusk/Dawn', or 'Unclear' (majority of the clip)>",
  "visual_or_infrared": "<'Visual' for normal RGB/color or grayscale footage, 'Infrared' if it looks like thermal/IR footage (false-color heat map, monochrome with heat signatures, etc.), or 'Unclear'>",
  "drone_type": "<'Fixed wing' if a fixed-wing drone/aircraft is visible anywhere in the clip, 'Rotary' if a multirotor/quadcopter/helicopter-style drone is visible, or 'NA' if no drone is visible or type cannot be determined>"
}

Rules:
- Base every answer ONLY on what is visibly present in the clip. Do not guess about things outside the video (like dataset licensing or which model it trains).
- If something is genuinely not visible or not determinable, use "NA", "Unclear", or "Not specified / NA" as appropriate rather than inventing an answer.
- Output must be strictly valid JSON, with double-quoted keys and string values, and nothing else in your response.
"""

IMAGE_USER_PROMPT = "Classify this drone-dataset sample image according to the schema in the system prompt. Respond with the JSON object only."
VIDEO_USER_PROMPT = "Classify this drone-dataset sample video clip according to the schema in the system prompt. Respond with the JSON object only."


# ------------------------------------------------------------------------
# Live media discovery
# ------------------------------------------------------------------------
def _api_request_with_retry(url, params, retries=2, backoff=1.5):
    last_exc = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, headers=AUTH_HEADERS, timeout=API_TIMEOUT)
            if r.status_code >= 500:
                raise requests.HTTPError(f"{r.status_code} Server Error", response=r)
            r.raise_for_status()
            return r
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status is not None and status < 500:
                raise
            last_exc = e
        except requests.RequestException as e:
            last_exc = e
        if attempt < retries:
            time.sleep(backoff * (attempt + 1))
    # pyrefly: ignore [bad-raise]
    raise last_exc


def get_splits(repo_id):
    r = _api_request_with_retry(f"{DATASETS_SERVER_BASE}/splits", {"dataset": repo_id})
    return r.json().get("splits", [])


def get_first_rows(repo_id, config, split):
    r = _api_request_with_retry(
        f"{DATASETS_SERVER_BASE}/first-rows",
        {"dataset": repo_id, "config": config, "split": split},
    )
    return r.json()


def _is_image_type(ftype):
    if isinstance(ftype, dict):
        if ftype.get("_type") == "Image":
            return True
        return any(_is_image_type(v) for v in ftype.values())
    if isinstance(ftype, list):
        return any(_is_image_type(item) for item in ftype)
    return False


def find_image_columns(features):
    return [feat["name"] for feat in features if _is_image_type(feat.get("type", {}))]


def extract_urls(rows, image_col):
    urls = []
    for entry in rows:
        val = entry.get("row", {}).get(image_col)
        if isinstance(val, dict) and "src" in val:
            urls.append(val["src"])
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, dict) and "src" in item:
                    urls.append(item["src"])
    return urls


def fetch_media_pool(repo_id):
    """Returns (image_urls, video_urls, error)."""
    image_urls = []
    video_urls = []
    error = ""
    found_images_via_api = False

    try:
        splits = get_splits(repo_id)
    except Exception as e:
        splits = []
        error = f"splits lookup failed: {e}"

    for split_info in splits:
        config = split_info.get("config", "default")
        split = split_info.get("split", "train")
        try:
            data = get_first_rows(repo_id, config, split)
        except Exception as e:
            error = f"first-rows failed for config={config} split={split}: {e}"
            continue
        image_cols = find_image_columns(data.get("features", []))
        if not image_cols:
            error = f"no Image-typed column in config={config} split={split}"
            continue
        urls = extract_urls(data.get("rows", []), image_cols[0])
        if urls:
            image_urls = urls
            found_images_via_api = True
            break
        error = f"Image column '{image_cols[0]}' found but no URLs resolved"

    if HfApi is not None:
        api = HfApi(token=HF_TOKEN)
        try:
            files = api.list_repo_files(repo_id, repo_type="dataset")
        except Exception as e:
            if not image_urls:
                error = f"file listing failed: {e}"
            files = []

        video_files = [
            f for f in files
            if f.lower().endswith(VIDEO_EXTENSIONS) and not any(p in f.lower() for p in NON_CONTENT_PATTERNS)
        ]
        if video_files:
            sample = random.sample(video_files, min(RAW_POOL_CAP, len(video_files)))
            video_urls = [f"https://huggingface.co/datasets/{repo_id}/resolve/main/{p}" for p in sample]

        if not found_images_via_api:
            image_files = [
                f for f in files
                if f.lower().endswith(IMAGE_EXTENSIONS) and not any(p in f.lower() for p in NON_CONTENT_PATTERNS)
            ]
            if image_files:
                sample = random.sample(image_files, min(RAW_POOL_CAP, len(image_files)))
                image_urls = [f"https://huggingface.co/datasets/{repo_id}/resolve/main/{p}" for p in sample]

    if image_urls and len(image_urls) > RAW_POOL_CAP:
        image_urls = random.sample(image_urls, RAW_POOL_CAP)

    if not image_urls and not video_urls:
        return [], [], (error or "no_image_data")

    return image_urls, video_urls, ""


# ------------------------------------------------------------------------
# Download (with curl fallback for CDN-level blocks)
# ------------------------------------------------------------------------
def _curl_download(url, dest_path):
    try:
        result = subprocess.run(
            [
                "curl", "-sS", "-L", "--fail",
                "--max-time", str(DOWNLOAD_TIMEOUT),
                "-A", DOWNLOAD_HEADERS["User-Agent"],
                "-H", f"Referer: {DOWNLOAD_HEADERS['Referer']}",
                "-o", dest_path,
                url,
            ],
            capture_output=True,
            timeout=DOWNLOAD_TIMEOUT + 5,
        )
        return result.returncode == 0 and os.path.exists(dest_path) and os.path.getsize(dest_path) > 0
    except Exception:
        return False


def download_media(url, dest_path, max_bytes):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = SESSION.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True)
            if resp.status_code in (401, 403, 404):
                return _curl_download(url, dest_path)
            resp.raise_for_status()

            content_length = resp.headers.get("Content-Length") or resp.headers.get("x-linked-size")
            if content_length and int(content_length) > max_bytes:
                print(f"skip ({int(content_length)/(1024*1024):.0f}MB > {max_bytes//(1024*1024)}MB limit)", end=" ", flush=True)
                return False

            downloaded = 0
            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=256 * 1024):
                    downloaded += len(chunk)
                    if downloaded > max_bytes:
                        print(f"skip (exceeded {max_bytes//(1024*1024)}MB while downloading)", end=" ", flush=True)
                        f.close()
                        if os.path.exists(dest_path):
                            os.remove(dest_path)
                        return False
                    f.write(chunk)
            return True
        except requests.exceptions.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status in (401, 403, 404):
                return _curl_download(url, dest_path)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_SEC)
        except Exception:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_SEC)
    return False


# ------------------------------------------------------------------------
# Video trimming (ffmpeg/ffprobe) - trims ONE clip per video, no per-frame
# extraction. The video backend's own processor handles frame sampling
# internally once it receives the clip.
# ------------------------------------------------------------------------
def get_video_info(video_path):
    """Returns dict with duration (s), fps, width, height - any of which
    may be None if ffprobe can't determine them."""
    info = {"duration": None, "fps": None, "width": None, "height": None}
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate,width,height:format=duration",
                "-of", "json", video_path,
            ],
            capture_output=True, text=True, timeout=30,
        )
        data = json.loads(result.stdout)
        fmt = data.get("format", {})
        streams = data.get("streams", [{}])
        stream = streams[0] if streams else {}

        duration = fmt.get("duration")
        if duration:
            info["duration"] = float(duration)

        fps_str = stream.get("r_frame_rate")
        if fps_str:
            num, den = fps_str.split("/")
            den_f = float(den)
            if den_f != 0:
                info["fps"] = float(num) / den_f

        if stream.get("width"):
            info["width"] = int(stream["width"])
        if stream.get("height"):
            info["height"] = int(stream["height"])
    except Exception:
        pass
    return info


def compute_window(duration, start_fraction, window_seconds):
    """Returns (start_time, actual_window_seconds), clamping to duration."""
    if not duration or duration <= 0:
        return 0.0, window_seconds
    start = duration * start_fraction
    remaining = duration - start
    if remaining >= window_seconds:
        return start, window_seconds
    if duration >= window_seconds:
        return max(0.0, duration - window_seconds), window_seconds
    return 0.0, duration


def extract_video_clip(video_path, out_path, start_fraction, window_seconds):
    """Trims ONE clip: seek to start_fraction into the video, take
    window_seconds. Tries a fast stream copy first, falls back to
    re-encoding if the copy fails (common with odd keyframe placement)."""
    info = get_video_info(video_path)
    start, window = compute_window(info["duration"], start_fraction, window_seconds)
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{start:.2f}", "-i", video_path, "-t", f"{window:.2f}", "-c", "copy", out_path],
            capture_output=True, timeout=120,
        )
        if result.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return True
        result = subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{start:.2f}", "-i", video_path, "-t", f"{window:.2f}", out_path],
            capture_output=True, timeout=300,
        )
        return result.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0
    except Exception:
        return False


# ------------------------------------------------------------------------
# CSV I/O
# ------------------------------------------------------------------------
def load_rows(input_path):
    with open(input_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames


def load_done_ids(output_path):
    done = set()
    if os.path.exists(output_path):
        with open(output_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                d_id = (row.get("dataset_id") or row.get("Dataset ID") or "").strip()
                if d_id:
                    done.add(d_id)
    return done


def ensure_output_header(output_path, fieldnames):
    if not os.path.exists(output_path):
        with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def append_row(output_path, row_dict, fieldnames):
    with open(output_path, "a", newline="", encoding="utf-8-sig") as f:
        csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore").writerow(row_dict)


# ------------------------------------------------------------------------
# Media -> content URL for the VLM request (local file:// path, cheap; or
# base64 data URI, portable but heavier)
# ------------------------------------------------------------------------
def media_to_content_url(path, kind, mode):
    """kind: 'image' or 'video'. mode: 'local-path' or 'base64'.

    For video on local-path mode: returns absolute file:// URL (requires
    the receiving server to be started with --allowed-local-media-path
    covering this path - true for vLLM, NOT true for mlx-omni-server/OMLX
    or most other local servers, which only accept base64/http(s) URLs).
    """
    if mode == "local-path":
        return f"file://{os.path.abspath(path)}"

    # base64 fallback (slow for video, use only if the server can't access
    # local paths - this is the default/required mode for images against
    # OMLX-style servers)
    ext = Path(path).suffix.lower().lstrip(".")
    if kind == "image":
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}" if ext else "image/jpeg"
    else:
        mime = f"video/{ext}" if ext else "video/mp4"

    # Warn on base64 video (expensive)
    if kind == "video":
        file_size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"    WARNING: base64-encoding {file_size_mb:.1f}MB video — this will be slow. "
              f"Use --video-media-mode local-path + vLLM --allowed-local-media-path instead.", flush=True)

    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


# ------------------------------------------------------------------------
# VLM calls
# ------------------------------------------------------------------------
def extract_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def extract_model_size(name):
    match = re.search(r"\b(\d+b)\b", name.lower())
    return match.group(1) if match else None


_RESOLVED_MODEL_CACHE = {}


def resolve_vlm_model_name(preferred_model=None, base_url=None, api_key=None):
    """Resolve a model name against a specific backend's /models list.
    Cache key includes the base_url so image and video backends (which may
    point at different servers) don't clobber each other's resolution."""
    base_url = base_url or VLM_BASE_URL
    api_key = api_key or VLM_API_KEY
    preferred = preferred_model or (VLM_MODEL if base_url == VLM_BASE_URL else None) or VLM_MODEL
    cache_key = (base_url, preferred)
    if cache_key in _RESOLVED_MODEL_CACHE:
        return _RESOLVED_MODEL_CACHE[cache_key]

    headers = {"Authorization": f"Bearer {api_key}"}
    resolved = preferred
    try:
        resp = requests.get(f"{base_url}/models", headers=headers, timeout=20)
        if resp.status_code == 200:
            models = resp.json().get("data", [])
            model_ids = [m.get("id") for m in models if isinstance(m, dict) and m.get("id")]
            if model_ids:
                if preferred in model_ids:
                    resolved = preferred
                else:
                    pref_lower = preferred.lower()
                    match = next((c for c in model_ids if pref_lower in c.lower() or c.lower() in pref_lower), None)
                    if match:
                        resolved = match
                    else:
                        pref_size = extract_model_size(preferred)
                        size_matched = [c for c in model_ids if pref_size and extract_model_size(c) == pref_size]
                        if size_matched:
                            resolved = size_matched[0]
                        elif preferred_model is not None or os.environ.get("VLM_MODEL") is not None:
                            raise RuntimeError(
                                f"Requested model '{preferred}' not available on {base_url}.\n"
                                f"Available models: {', '.join(model_ids)}"
                            )
                        else:
                            resolved = model_ids[0]
    except RuntimeError:
        raise
    except Exception:
        pass

    _RESOLVED_MODEL_CACHE[cache_key] = resolved
    return resolved


def _call_vlm(content_block, system_prompt, preferred_model=None, timeout=REQUEST_TIMEOUT,
              is_video=False, base_url=None, api_key=None):
    base_url = base_url or VLM_BASE_URL
    api_key = api_key or VLM_API_KEY
    resolved_model = resolve_vlm_model_name(preferred_model, base_url=base_url, api_key=api_key)
    payload = {
        "model": resolved_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_block},
        ],
        "temperature": 0.1,
        "max_tokens": 500,
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    last_err = None
    last_status = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(f"{base_url}/chat/completions", json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return extract_json(content)
        except requests.HTTPError as e:
            last_err = e
            last_status = getattr(e.response, "status_code", None)
            try:
                print(f"    [VLM 400 body] {e.response.text[:500]}", flush=True)
            except Exception:
                pass
            if is_video and last_status in (400, 415, 422):
                raise
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_SEC)
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_SEC)
    raise RuntimeError(f"VLM call failed after {MAX_RETRIES} attempts: {last_err}")


def call_vlm_image(image_path, media_mode, preferred_model=None, base_url=None, api_key=None):
    content_url = media_to_content_url(image_path, "image", media_mode)
    content_block = [
        {"type": "text", "text": IMAGE_USER_PROMPT},
        {"type": "image_url", "image_url": {"url": content_url}},
    ]
    return _call_vlm(content_block, IMAGE_SYSTEM_PROMPT, preferred_model, base_url=base_url, api_key=api_key)


def call_vlm_video(video_path, media_mode, preferred_model=None, base_url=None, api_key=None):
    content_url = media_to_content_url(video_path, "video", media_mode)
    content_block = [
        {"type": "text", "text": VIDEO_USER_PROMPT},
        {"type": "video_url", "video_url": {"url": content_url}},
    ]
    try:
        return _call_vlm(content_block, VIDEO_SYSTEM_PROMPT, preferred_model, timeout=REQUEST_TIMEOUT,
                          is_video=True, base_url=base_url, api_key=api_key)
    except requests.HTTPError as e:
        # _call_vlm re-raises HTTPError as-is (no retry) for 400/415/422 on
        # video calls, since a bad request/unsupported-media-type response
        # means the backend doesn't accept video input at all - retrying
        # won't help.
        status = getattr(e.response, "status_code", None)
        if status in (400, 415, 422):
            raise ValueError(f"Video format not supported by this model: {e}")
        raise
    except RuntimeError as e:
        if "400" in str(e) or "415" in str(e) or "422" in str(e):
            raise ValueError(f"Video format not supported by this model: {e}")
        raise


# ------------------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------------------
IGNORE_TOKENS = {"na", "n/a", "unclear", "not specified / na", "not specified", ""}


def vote(values):
    """Returns (label, count, total) where count is the number of usable
    answers matching the winning label, and total is the number of
    non-empty answers considered."""
    cleaned = [v.strip() for v in values if isinstance(v, str) and v.strip()]
    usable = [v for v in cleaned if v.strip().lower() not in IGNORE_TOKENS]
    total = len(cleaned)
    if not usable:
        return ("NA (no usable answers)", 0, total) if total == 0 else ("Unclear", total, total)
    top_label, top_count = Counter(usable).most_common(1)[0]
    return top_label, top_count, total


def vote_any_yes(values):
    """Returns (label, count, total)."""
    cleaned = [v.strip() for v in values if isinstance(v, str) and v.strip()]
    usable = [v for v in cleaned if v.strip().lower() not in IGNORE_TOKENS]
    total = len(cleaned)
    if not usable:
        return "Not specified / NA", 0, total
    yes_count = sum(1 for v in usable if v.strip().lower() == "yes")
    if yes_count > 0:
        return "Yes", yes_count, total
    no_count = sum(1 for v in usable if v.strip().lower() == "no")
    if no_count > 0:
        return "No", 0, total
    return "Not specified / NA", 0, total


def combine_drone_types(values):
    """Returns (label, count, total). For 'Both', count is the Rotary
    count and a second (fixed-wing) count is embedded in the label text
    since this field can genuinely be a mix rather than a single winner."""
    cleaned = [v.strip() for v in values if isinstance(v, str) and v.strip()]
    usable = {v for v in cleaned if v.strip().lower() not in IGNORE_TOKENS}
    total = len(cleaned)
    if not usable:
        return "NA (no drone visible in sampled items)", 0, total
    rotary_n = sum(1 for v in cleaned if v == "Rotary")
    fixed_n = sum(1 for v in cleaned if v == "Fixed wing")
    if usable == {"Rotary"}:
        return "Rotary", rotary_n, total
    if usable == {"Fixed wing"}:
        return "Fixed wing", fixed_n, total
    return f"Both (Rotary {rotary_n}, Fixed wing {fixed_n})", rotary_n + fixed_n, total


def extract_license_info(input_row):
    if not input_row:
        return "NA"
    lic = (input_row.get("License") or input_row.get("license") or "").strip()
    if not lic or lic.lower() in ("none", "null", "undefined"):
        tags = str(input_row.get("Tags") or input_row.get("tags") or "")
        match = re.search(r"license:([a-zA-Z0-9._-]+)", tags)
        if match:
            lic = match.group(1).strip()
    if lic and lic.lower() not in ("none", "null", "undefined", ""):
        return f"Free ({lic})"
    return "NA"


def extract_updated_info(input_row):
    if not input_row:
        return "NA"
    last_mod = (
        input_row.get("Last modified")
        or input_row.get("last_modified")
        or input_row.get("Created date")
        or ""
    ).strip()
    if last_mod:
        return last_mod.split("T")[0] if "T" in last_mod else last_mod
    return "NA"


def compute_majority_resolution(resolutions):
    """Returns (label, count, total)."""
    if not resolutions:
        return "NA", 0, 0
    total = len(resolutions)
    top_res, top_count = Counter(resolutions).most_common(1)[0]
    return top_res, top_count, total


def build_aggregated_row(dataset_id, results, n_sampled, resolutions=None, input_row=None, media_summary=""):
    row = {col: "NA" for col in OUTPUT_COLUMNS}
    row["dataset_id"] = dataset_id
    row["Availability and licensing (free/which license)"] = extract_license_info(input_row)
    row["How recently updated?"] = extract_updated_info(input_row)

    res_label, res_count, res_total = compute_majority_resolution(resolutions)
    row["Resolution of images (majority)"] = res_label
    row["Resolution of images (majority) - count"] = res_count
    row["Resolution of images (majority) - total"] = res_total

    if not results:
        row["Image / Video"] = f"NA (VLM failed on all {n_sampled} sampled items{(' - ' + media_summary) if media_summary else ''})"
        for field in DETERMINISTIC_NA_FIELDS:
            row[field] = "NA"
        return row

    n_ok = len(results)
    descriptions = [r.get("image_description", "") for r in results if r.get("image_description")]
    rep_desc = descriptions[0] if descriptions else "NA"
    summary_part = f" [{media_summary}]" if media_summary else ""
    row["Image / Video"] = f"[{n_ok}/{n_sampled} sampled{summary_part}] Example: {rep_desc}"

    fd_label, fd_count, fd_total = vote([r.get("from_drone_or_of_drone", "NA") for r in results])
    row["From drone / Of drone"] = fd_label
    row["From drone / Of drone - count"] = fd_count
    row["From drone / Of drone - total"] = fd_total

    cd_label, cd_count, cd_total = vote([r.get("closeup_or_distant", "NA") for r in results])
    row["Within dataset - do you have more closeups or more distant shots (of drones)"] = cd_label
    row["Within dataset - do you have more closeups or more distant shots (of drones) - count"] = cd_count
    row["Within dataset - do you have more closeups or more distant shots (of drones) - total"] = cd_total

    nd_label, nd_count, nd_total = vote_any_yes([r.get("non_drone_flying_objects", "NA") for r in results])
    row["Does dataset also have non-drone flying objects (planes/birds)"] = nd_label
    row["Does dataset also have non-drone flying objects (planes/birds) - count"] = nd_count
    row["Does dataset also have non-drone flying objects (planes/birds) - total"] = nd_total

    dn_label, dn_count, dn_total = vote([r.get("day_or_night", "NA") for r in results])
    row["Day or night"] = dn_label
    row["Day or night - count"] = dn_count
    row["Day or night - total"] = dn_total

    vi_label, vi_count, vi_total = vote([r.get("visual_or_infrared", "NA") for r in results])
    row["Visual or Infra red images"] = vi_label
    row["Visual or Infra red images - count"] = vi_count
    row["Visual or Infra red images - total"] = vi_total

    dt_label, dt_count, dt_total = combine_drone_types([r.get("drone_type", "NA") for r in results])
    row["Type of drone (fixed wing / rotary)"] = dt_label
    row["Type of drone (fixed wing / rotary) - count"] = dt_count
    row["Type of drone (fixed wing / rotary) - total"] = dt_total

    for field in DETERMINISTIC_NA_FIELDS:
        row[field] = "NA"

    return row


def validate_setup(args):
    """Check backend connectivity/config before starting the full run."""
    print("Validating setup...", flush=True)

    # Check image backend is reachable
    try:
        resp = requests.get(f"{args.vlm_base_url}/models", headers={"Authorization": f"Bearer {args.vlm_api_key}"}, timeout=10)
        if resp.status_code != 200:
            print(f"  WARNING: image backend /models returned {resp.status_code} at {args.vlm_base_url}. Check API key/URL.", flush=True)
            return False
        models = resp.json().get("data", [])
        available_model = next((m.get("id") for m in models if isinstance(m, dict)), None)
        if not available_model:
            print(f"  ERROR: No models available on image backend at {args.vlm_base_url}", flush=True)
            return False
        print(f"  ✓ Image backend reachable ({args.vlm_base_url}). Available models: {[m.get('id') for m in models if isinstance(m, dict)]}", flush=True)
    except Exception as e:
        print(f"  ERROR: Cannot reach image backend at {args.vlm_base_url}: {e}", flush=True)
        return False

    # Check video backend only if it's a *different* server from the image one
    if args.video_vlm_base_url and args.video_vlm_base_url != args.vlm_base_url:
        try:
            resp = requests.get(
                f"{args.video_vlm_base_url}/models",
                headers={"Authorization": f"Bearer {args.video_vlm_api_key}"}, timeout=10,
            )
            if resp.status_code != 200:
                print(f"  WARNING: video backend /models returned {resp.status_code} at {args.video_vlm_base_url}. "
                      f"Videos will fail/skip until this is reachable; images are unaffected.", flush=True)
            else:
                models = resp.json().get("data", [])
                print(f"  ✓ Video backend reachable ({args.video_vlm_base_url}). Available models: "
                      f"{[m.get('id') for m in models if isinstance(m, dict)]}", flush=True)
        except Exception as e:
            print(f"  WARNING: Cannot reach video backend at {args.video_vlm_base_url}: {e}. "
                  f"Videos will fail/skip until this is reachable; images are unaffected.", flush=True)
    else:
        print(f"  (no separate video backend configured — videos will use the image backend, "
              f"mode={args.video_media_mode})", flush=True)

    # Check local-path mode permissions if needed (only relevant to video path)
    if args.video_media_mode == "local-path":
        abs_dir = os.path.abspath(args.images_dir)
        if not os.path.exists(abs_dir):
            os.makedirs(abs_dir, exist_ok=True)
            print(f"  ✓ Created --images-dir: {abs_dir}", flush=True)
        else:
            print(f"  ✓ --images-dir writable: {abs_dir}", flush=True)

    # Check ffmpeg/ffprobe available
    for tool in ("ffmpeg", "ffprobe"):
        try:
            subprocess.run([tool, "-version"], capture_output=True, timeout=5)
            print(f"  ✓ {tool} found on PATH", flush=True)
        except Exception:
            print(f"  ERROR: {tool} not found on PATH. Install ffmpeg before running.", flush=True)
            return False

    print("Setup validation passed.\n", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--id-column", default="dataset_id")
    parser.add_argument(
        "--n-images", type=int, default=DEFAULT_N_IMAGES,
        help=f"Max still-IMAGE samples per dataset (default {DEFAULT_N_IMAGES}). Does not affect video handling.",
    )
    parser.add_argument("--video-start-fraction", type=float, default=DEFAULT_VIDEO_START_FRACTION,
                         help="Seek this fraction into each video before trimming the clip (default 0.20)")
    parser.add_argument("--video-window-seconds", type=float, default=DEFAULT_VIDEO_WINDOW_SECONDS,
                         help="Length of the trimmed clip sent to the model per video (default 30s)")
    parser.add_argument("--max-videos", type=int, default=None,
                         help="Optional cap on number of videos processed per dataset (default: unlimited)")

    # --- Early stop on "unclear" datasets (saves compute on junk/ambiguous data) ---
    parser.add_argument("--early-stop-check-n", type=int, default=20,
                         help="After this many IMAGES are classified for a dataset, check the 'Unclear' rate "
                              "on from_drone_or_of_drone and stop sampling further images if it's too high "
                              "(default 20). Set to 0 to disable early stopping entirely.")
    parser.add_argument("--early-stop-unclear-ratio", type=float, default=0.5,
                         help="If the fraction of Unclear/NA answers on from_drone_or_of_drone in the first "
                              "--early-stop-check-n images is >= this ratio, stop sampling more images for "
                              "this dataset (default 0.5 = 50%%).")

    # --- Image backend (default: whatever local server you already run) ---
    parser.add_argument("--vlm-base-url", default=VLM_BASE_URL,
                         help=f"Base URL for the IMAGE backend (default: {VLM_BASE_URL} / $VLM_BASE_URL)")
    parser.add_argument("--vlm-model", default=None,
                         help="Model name for the image backend (default: $VLM_MODEL or a sensible fallback)")
    parser.add_argument("--vlm-api-key", default=VLM_API_KEY,
                         help="API key for the image backend (default: $VLM_API_KEY/$OPENAI_API_KEY)")
    parser.add_argument("--image-media-mode", choices=["local-path", "base64"], default="base64",
                         help="How images are sent to the image backend. Default 'base64' - works against "
                              "mlx-omni-server/OMLX and most local servers without extra flags. Use 'local-path' "
                              "only if your image backend is vLLM with --allowed-local-media-path set.")

    # --- Video backend (optional separate server, e.g. vLLM on Linux) ---
    parser.add_argument("--video-vlm-base-url", default=VIDEO_VLM_BASE_URL,
                         help="Base URL for the VIDEO backend. If unset, videos use --vlm-base-url instead "
                              "(no separate server required). Set this to your vLLM instance to offload "
                              "video-only to vLLM while keeping images on your current server.")
    parser.add_argument("--video-vlm-model", default=VIDEO_VLM_MODEL,
                         help="Model name for the video backend (default: $VIDEO_VLM_MODEL, or falls back to "
                              "--vlm-model if no separate video backend is set)")
    parser.add_argument("--video-vlm-api-key", default=VIDEO_VLM_API_KEY,
                         help="API key for the video backend (default: $VIDEO_VLM_API_KEY, or falls back to "
                              "--vlm-api-key if no separate video backend is set)")
    parser.add_argument("--video-media-mode", choices=["local-path", "base64"], default="local-path",
                         help="How video clips are sent to the video backend. Default 'local-path' (cheap, "
                              "requires vLLM --allowed-local-media-path covering --images-dir). Use 'base64' "
                              "if the video backend can't access local paths.")

    parser.add_argument("--max-image-mb", type=int, default=DEFAULT_MAX_IMAGE_MB)
    parser.add_argument("--max-video-mb", type=int, default=DEFAULT_MAX_VIDEO_MB)
    parser.add_argument("--images-dir", default="dataset_images")
    parser.add_argument("--limit", type=int, default=None, help="Cap number of DATASETS processed this run")
    parser.add_argument("--skip-validation", action="store_true", help="Skip setup validation (for resume/testing)")
    keep_group = parser.add_mutually_exclusive_group()
    keep_group.add_argument("--keep-videos", dest="keep_videos", action="store_true")
    keep_group.add_argument("--no-keep-videos", dest="keep_videos", action="store_false")
    parser.set_defaults(keep_videos=False)  # Default to cleanup (save disk)
    args = parser.parse_args()

    # If no separate video backend was given, video calls transparently
    # fall back to the image backend/model/key - no vLLM required.
    if not args.video_vlm_base_url:
        args.video_vlm_base_url = args.vlm_base_url
        args.video_vlm_model = args.video_vlm_model or args.vlm_model
        args.video_vlm_api_key = args.video_vlm_api_key or args.vlm_api_key
        # also fall back the media mode: if the user didn't explicitly ask
        # for local-path video and there's no dedicated vLLM box, base64 is
        # far more likely to actually work against the same local server.
        if args.video_media_mode == "local-path" and args.video_vlm_base_url == args.vlm_base_url:
            print("No --video-vlm-base-url set — videos will use the same backend as images "
                  f"({args.vlm_base_url}). Forcing --video-media-mode to 'base64' since local file:// "
                  "paths generally aren't supported outside vLLM.", flush=True)
            args.video_media_mode = "base64"

    # Validate before committing time/disk
    if not args.skip_validation:
        if not validate_setup(args):
            print("Setup validation failed. Exiting.")
            return

    args.images_dir = os.path.abspath(args.images_dir)
    os.makedirs(args.images_dir, exist_ok=True)
    rows, _ = load_rows(args.input)

    ensure_output_header(args.output, OUTPUT_COLUMNS)
    done_ids = load_done_ids(args.output)
    print(
        f"Already done: {len(done_ids)} datasets. Processing {len(rows) - len(done_ids)} remaining... "
        f"(n_images={args.n_images}, video_window={args.video_window_seconds}s @ {int(args.video_start_fraction*100)}% mark, "
        f"image_backend={args.vlm_base_url} [{args.image_media_mode}], "
        f"video_backend={args.video_vlm_base_url} [{args.video_media_mode}], keep_videos={args.keep_videos})"
    )
    if args.video_media_mode == "local-path":
        print(f"  Media path (video backend must have --allowed-local-media-path covering this): {args.images_dir}")

    processed = 0
    # Once the video backend proves it can't handle video (400/415/422 on a
    # real attempt), stop downloading+trimming further clips this run —
    # there's no point burning bandwidth/disk on calls we know will fail.
    video_backend_unsupported = {"flag": False}

    for i, row in enumerate(rows):
        dataset_id = (row.get(args.id_column) or row.get("Dataset ID") or row.get("dataset_id") or "").strip()
        if not dataset_id or dataset_id in done_ids:
            continue

        print(f"[{i+1}/{len(rows)}] {dataset_id}: discovering media...", flush=True)
        image_urls, video_urls, fetch_error = fetch_media_pool(dataset_id)

        if not image_urls and not video_urls:
            out_row = build_aggregated_row(dataset_id, [], 0, input_row=row)
            out_row["Image / Video"] = f"NA (live media discovery failed: {fetch_error})"
            append_row(args.output, out_row, OUTPUT_COLUMNS)
            done_ids.add(dataset_id)
            processed += 1
            print(f"    no media found: {fetch_error}", flush=True)
            continue

        sampled_images = random.sample(image_urls, min(args.n_images, len(image_urls))) if image_urls else []
        sampled_videos = video_urls
        if args.max_videos is not None and len(sampled_videos) > args.max_videos:
            sampled_videos = random.sample(sampled_videos, args.max_videos)

        print(f"    plan: {len(sampled_images)} images, {len(sampled_videos)} videos "
              f"(each video = 1 clip, 1 classification call)", flush=True)

        safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", dataset_id)
        dataset_img_dir = os.path.join(args.images_dir, safe_name)
        os.makedirs(dataset_img_dir, exist_ok=True)
        videos_dir = os.path.join(dataset_img_dir, "videos")

        results = []
        resolutions = []
        n_sampled_total = 0
        stopped_early = False

        if video_backend_unsupported["flag"] and sampled_videos:
            print(f"    skipping {len(sampled_videos)} videos without downloading — video backend already "
                  f"confirmed unsupported this run", flush=True)

        # Parallel image download (network I/O bound)
        def process_image(args_tuple):
            j, url, dataset_img_dir, media_mode, vlm_model, vlm_base_url, vlm_api_key = args_tuple
            ext = Path(url.split("?")[0]).suffix or ".jpg"
            if not ext.startswith("."):
                ext = f".{ext}"
            img_path = os.path.join(dataset_img_dir, f"img{j:04d}{ext}")

            if not download_media(url, img_path, args.max_image_mb * 1024 * 1024):
                return None

            # Downscale before sending — a full-resolution frame (e.g.
            # 1920x1080) base64-encodes into thousands of tokens and is
            # what was blowing past the server's context limit. 1024px on
            # the long side is plenty for the classification fields we ask
            # for and cuts payload size drastically.
            try:
                _downscale_for_vlm(img_path)
            except Exception:
                pass  # if downscaling fails for some reason, fall back to the original file

            res_str = None
            try:
                with Image.open(img_path) as im:
                    res_str = f"{im.size[0]}x{im.size[1]}"
            except Exception:
                pass

            try:
                vlm_result = call_vlm_image(
                    img_path, media_mode, preferred_model=vlm_model,
                    base_url=vlm_base_url, api_key=vlm_api_key,
                )
                return (vlm_result, res_str)
            except RuntimeError as e:
                # A single bad/oversized/crashed-server image should never
                # take down a multi-hour, multi-dataset run — log it and
                # skip this one image instead of propagating.
                print(f"VLM failed (skipping this image): {e}", flush=True)
                return None
            except Exception as e:
                print(f"VLM failed: {e}", flush=True)
                return None

        def run_image_batch(url_batch, start_index):
            """Downloads+classifies one batch of images, appending into
            results/resolutions and returning how many were attempted."""
            nonlocal n_sampled_total
            attempted = 0
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    executor.submit(
                        process_image,
                        (start_index + j, url, dataset_img_dir, args.image_media_mode, args.vlm_model, args.vlm_base_url, args.vlm_api_key),
                    ): j
                    for j, url in enumerate(url_batch)
                }
                for future in as_completed(futures):
                    n_sampled_total += 1
                    attempted += 1
                    try:
                        result = future.result()
                        if result:
                            vlm_result, res_str = result
                            results.append(vlm_result)
                            if res_str:
                                resolutions.append(res_str)
                            print(f"    [image {n_sampled_total}/{len(sampled_images)}] ok", flush=True)
                        else:
                            print(f"    [image {n_sampled_total}/{len(sampled_images)}] skipped", flush=True)
                    except Exception as e:
                        # Belt-and-suspenders: process_image already swallows
                        # per-image failures and returns None, but if
                        # anything unexpected still escapes, log and skip
                        # rather than crashing the whole multi-dataset run.
                        print(f"    [image {n_sampled_total}/{len(sampled_images)}] error (skipping): {e}", flush=True)
            return attempted

        if sampled_images:
            check_n = args.early_stop_check_n
            early_stop_enabled = check_n and check_n > 0

            if early_stop_enabled and len(sampled_images) > check_n:
                checkpoint_batch = sampled_images[:check_n]
                remaining_batch = sampled_images[check_n:]
            else:
                checkpoint_batch = sampled_images
                remaining_batch = []

            print(f"    Downloading + classifying {len(sampled_images)} images...", flush=True)
            run_image_batch(checkpoint_batch, 0)

            if early_stop_enabled and remaining_batch:
                checkpoint_results = results[-len(checkpoint_batch):] if len(checkpoint_batch) <= len(results) else results
                unclear_count = sum(
                    1 for r in checkpoint_results
                    if str(r.get("from_drone_or_of_drone", "")).strip().lower() in ("unclear", "na", "n/a", "")
                )
                unclear_ratio = unclear_count / len(checkpoint_batch) if checkpoint_batch else 0
                if unclear_ratio >= args.early_stop_unclear_ratio:
                    stopped_early = True
                    print(f"    early stop: {unclear_count}/{len(checkpoint_batch)} "
                          f"({unclear_ratio:.0%}) of checkpoint images came back Unclear/NA on "
                          f"from_drone_or_of_drone (threshold {args.early_stop_unclear_ratio:.0%}) — "
                          f"skipping remaining {len(remaining_batch)} images for this dataset", flush=True)

            if remaining_batch and not stopped_early:
                run_image_batch(remaining_batch, len(checkpoint_batch))

        for k, url in enumerate(sampled_videos):
            if video_backend_unsupported["flag"]:
                # Already confirmed this run that the video backend rejects
                # video input — don't download/trim/send any more clips.
                break

            n_sampled_total += 1
            ext = Path(url.split("?")[0]).suffix or ".mp4"
            if not ext.startswith("."):
                ext = f".{ext}"
            os.makedirs(videos_dir, exist_ok=True)
            video_path = os.path.join(videos_dir, f"vid{k:03d}{ext}")
            clip_path = os.path.join(videos_dir, f"vid{k:03d}_clip{ext}")

            print(f"    [video {k+1}/{len(sampled_videos)}] downloading...", end=" ", flush=True)
            if not download_media(url, video_path, args.max_video_mb * 1024 * 1024):
                print("download failed", flush=True)
                continue

            print(f"trimming {args.video_window_seconds}s clip...", end=" ", flush=True)
            if not extract_video_clip(video_path, clip_path, args.video_start_fraction, args.video_window_seconds):
                print("clip extraction failed", flush=True)
                if not args.keep_videos:
                    try:
                        os.remove(video_path)
                    except OSError:
                        pass
                continue

            info = get_video_info(clip_path)
            if info["width"] and info["height"]:
                resolutions.append(f"{info['width']}x{info['height']}")

            print(f"sending whole clip to video backend ({args.video_vlm_base_url})...", end=" ", flush=True)
            try:
                results.append(call_vlm_video(
                    clip_path, args.video_media_mode, preferred_model=args.video_vlm_model,
                    base_url=args.video_vlm_base_url, api_key=args.video_vlm_api_key,
                ))
                print("ok", flush=True)
            except ValueError as e:
                # Model doesn't support video (e.g., MLX/omni server without vLLM).
                # Confirmed once — stop attempting further videos this run.
                video_backend_unsupported["flag"] = True
                print(f"skipped (video not supported by this backend: {e})", flush=True)
                print(f"    → video backend at {args.video_vlm_base_url} does not accept video input. "
                      f"Remaining videos for this and future datasets will be skipped without downloading. "
                      f"Point --video-vlm-base-url at a vLLM instance to enable video classification.", flush=True)
            except RuntimeError as e:
                print(f"VLM failed (skipping this video): {e}", flush=True)
            except Exception as e:
                print(f"VLM failed: {e}", flush=True)
            finally:
                # Clean up immediately after send (whether success or fail) to free disk
                if not args.keep_videos:
                    for p in (video_path, clip_path):
                        try:
                            os.remove(p)
                        except OSError:
                            pass

        early_stop_note = " - early stopped, high Unclear rate" if stopped_early else ""
        media_summary = f"{len(sampled_images)} images planned{early_stop_note}, {len(sampled_videos)} videos (whole-clip)"
        out_row = build_aggregated_row(
            dataset_id, results, n_sampled_total, resolutions=resolutions, input_row=row, media_summary=media_summary
        )
        append_row(args.output, out_row, OUTPUT_COLUMNS)
        done_ids.add(dataset_id)
        processed += 1
        print(f"    done: {len(results)}/{n_sampled_total} items classified successfully.", flush=True)

        if args.limit and processed >= args.limit:
            print(f"Hit --limit {args.limit} datasets, stopping.")
            break

    print(f"Done. Processed {processed} new datasets this run. Total done: {len(done_ids)}.")


if __name__ == "__main__":
    main()