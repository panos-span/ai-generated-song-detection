"""Download FakeMusicCaps dataset from Zenodo and build metadata.

Downloads ``FakeMusicCaps.zip`` (~12.9 GB) from Zenodo via the REST API,
extracts model folders (excluding the ``MusicCaps/`` real-audio folder),
re-saves WAV files as 16 kHz mono PCM_16, and matches captions from the
MusicCaps CSV hosted on HuggingFace.

Usage::

    python data/download_fakemusiccaps.py --num_samples 200 --output_dir data/fakemusiccaps
"""
from __future__ import annotations

import argparse
import logging
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
import soundfile as sf
import urllib3
from tqdm import tqdm

# Corporate proxies may inject certs that break SSL verification
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

TARGET_SR = 16000

# AI-generated model folders inside the zip — order preserved for consistency.
VALID_MODELS: set[str] = {
    "MusicGen_medium",
    "musicldm",
    "audioldm2",
    "stable_audio_open",
    "mustango",
}

MUSICCAPS_CSV_URL = (
    "https://huggingface.co/datasets/google/MusicCaps/resolve/main/musiccaps-public.csv"
)

ZENODO_API_URL = "https://zenodo.org/api/records"


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download FakeMusicCaps dataset from Zenodo",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=200,
        help="Total number of tracks to process across all models",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/fakemusiccaps",
        help="Output directory (default: data/fakemusiccaps)",
    )
    parser.add_argument(
        "--zenodo_record",
        type=str,
        default="15063698",
        help="Zenodo record ID for FakeMusicCaps (default: 15063698)",
    )
    return parser.parse_args()


# ======================================================================
# Metadata helpers
# ======================================================================

def build_metadata_row(filename: str, model: str, caption: str) -> dict:
    """Build a single metadata row."""
    return {
        "filename": filename,
        "model": model,
        "caption": caption,
    }


def get_existing_filenames(csv_path: Path) -> set[str]:
    """Return filenames already recorded in *csv_path*, or an empty set."""
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    return set(df["filename"].tolist())


def is_valid_model(name: str) -> bool:
    """Return True if *name* is a known AI-model folder (not MusicCaps)."""
    return name in VALID_MODELS


# ======================================================================
# Zenodo download
# ======================================================================

def get_zenodo_download_url(record_id: str) -> tuple[str, int]:
    """Query the Zenodo REST API and return (download_url, file_size).

    Raises ``SystemExit`` on failure.
    """
    api_url = f"{ZENODO_API_URL}/{record_id}"
    try:
        resp = requests.get(api_url, timeout=30, verify=False)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.error("Zenodo API request failed: %s", exc)
        raise SystemExit(1) from exc

    data = resp.json()
    files = data.get("files", [])
    if not files:
        logger.error("No files found in Zenodo record %s", record_id)
        raise SystemExit(1)

    # Find the zip file
    for f in files:
        if f["key"].endswith(".zip"):
            url = f["links"]["self"]
            size = f["size"]
            return url, size

    # Fallback to first file
    first = files[0]
    return first["links"]["self"], first["size"]


def download_zip(url: str, dest: Path, expected_size: int) -> None:
    """Stream-download a file with tqdm progress and resume support.

    Skips the download if *dest* already exists with the correct size.
    Resumes from the existing byte offset if a partial file is present.
    """
    if dest.exists():
        local_size = dest.stat().st_size
        if local_size == expected_size:
            logger.info("Zip already downloaded (%d bytes), skipping.", local_size)
            return
        logger.info(
            "Partial download found (%d / %d bytes), resuming.",
            local_size,
            expected_size,
        )
    else:
        local_size = 0

    dest.parent.mkdir(parents=True, exist_ok=True)

    headers = {}
    if local_size > 0:
        headers["Range"] = f"bytes={local_size}-"

    try:
        resp = requests.get(url, stream=True, timeout=60, verify=False, headers=headers)
        # 206 = Partial Content (server supports resume), 200 = full restart
        if local_size > 0 and resp.status_code == 200:
            logger.warning("Server does not support range requests — restarting from 0.")
            local_size = 0
        elif resp.status_code not in (200, 206):
            resp.raise_for_status()
    except requests.RequestException as exc:
        logger.error("Download failed: %s", exc)
        raise SystemExit(1) from exc

    total = int(resp.headers.get("Content-Length", expected_size - local_size)) + local_size
    write_mode = "ab" if local_size > 0 and resp.status_code == 206 else "wb"
    logger.info("Downloading %s (%d bytes)...", dest.name, expected_size)

    with open(dest, write_mode) as fh, tqdm(
        total=total, initial=local_size, unit="B", unit_scale=True, desc=dest.name,
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MB chunks
            fh.write(chunk)
            pbar.update(len(chunk))

    logger.info("Download complete: %s", dest)


# ======================================================================
# Extraction
# ======================================================================

def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    """Extract *zip_path* into *extract_dir*."""
    if extract_dir.exists() and any(extract_dir.iterdir()):
        logger.info("Extraction directory already exists, skipping extraction.")
        return
    extract_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting %s...", zip_path.name)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    logger.info("Extraction complete.")


def discover_model_folders(extract_dir: Path) -> list[tuple[str, Path]]:
    """Walk *extract_dir* and return ``(model_name, folder_path)`` pairs.

    Only returns folders whose name is in :data:`VALID_MODELS` (excludes
    ``MusicCaps/`` and the macOS ``__MACOSX/`` artefact directory).
    Returns only the shallowest match per model name to avoid duplicates
    caused by ``__MACOSX/<model>/`` shadows of the real folders.
    """
    seen: dict[str, Path] = {}
    for item in sorted(extract_dir.rglob("*")):
        # Skip anything inside a __MACOSX directory
        if "__MACOSX" in item.parts:
            continue
        if item.is_dir() and item.name in VALID_MODELS:
            if item.name not in seen:
                seen[item.name] = item
    return sorted(seen.items())


# ======================================================================
# MusicCaps caption matching
# ======================================================================

def download_musiccaps_csv(staging_dir: Path) -> pd.DataFrame:
    """Download and cache the MusicCaps CSV for caption look-up."""
    csv_path = staging_dir / "musiccaps-public.csv"
    if csv_path.exists():
        logger.info("MusicCaps CSV already cached at %s", csv_path)
        return pd.read_csv(csv_path)

    logger.info("Downloading MusicCaps CSV...")
    try:
        resp = requests.get(MUSICCAPS_CSV_URL, timeout=60, verify=False)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning(
            "Failed to download MusicCaps CSV: %s. Captions will be empty.", exc,
        )
        return pd.DataFrame(columns=["ytid", "caption"])

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_bytes(resp.content)
    return pd.read_csv(csv_path)


def build_caption_lookup(musiccaps_df: pd.DataFrame) -> dict[str, str]:
    """Build a ``{ytid: caption}`` mapping from the MusicCaps DataFrame."""
    lookup: dict[str, str] = {}
    if "ytid" not in musiccaps_df.columns or "caption" not in musiccaps_df.columns:
        return lookup
    for _, row in musiccaps_df.iterrows():
        ytid = str(row["ytid"]).strip()
        caption = str(row["caption"]).strip() if pd.notna(row["caption"]) else ""
        if ytid:
            lookup[ytid] = caption
    return lookup


# ======================================================================
# Audio re-save
# ======================================================================

def resave_wav_pcm16(src_path: Path, dst_path: Path) -> bool:
    """Read a WAV file and re-save as 16 kHz mono PCM_16.

    The source FakeMusicCaps files are float32; this converts to PCM_16 for
    project consistency.
    """
    try:
        data, sr = sf.read(str(src_path), dtype="float32")
        # Ensure mono
        if data.ndim > 1:
            data = data.mean(axis=1)
        sf.write(str(dst_path), data, sr, subtype="PCM_16")
        return True
    except Exception as e:
        logger.warning("Error re-saving %s: %s", src_path.name, e)
        return False


def _resave_worker(args: tuple) -> dict | None:
    """Thread worker: re-save one WAV and return its metadata row."""
    src_path, dst_path, metadata_row = args
    ok = resave_wav_pcm16(src_path, dst_path)
    return metadata_row if ok else None


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
    )

    output_dir = Path(args.output_dir)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metadata.csv"

    existing_files = get_existing_filenames(csv_path)
    if existing_files:
        logger.info("Found %d already-processed files, will skip them", len(existing_files))

    # ------------------------------------------------------------------
    # 1. Get Zenodo download URL
    # ------------------------------------------------------------------
    download_url, file_size = get_zenodo_download_url(args.zenodo_record)
    logger.info("Zenodo download URL: %s (%d bytes)", download_url, file_size)

    # ------------------------------------------------------------------
    # 2. Download the zip
    # ------------------------------------------------------------------
    staging_dir = output_dir / "_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    zip_path = staging_dir / "FakeMusicCaps.zip"
    download_zip(download_url, zip_path, file_size)

    # ------------------------------------------------------------------
    # 3. Extract
    # ------------------------------------------------------------------
    extract_dir = staging_dir / "extracted"
    extract_zip(zip_path, extract_dir)

    # ------------------------------------------------------------------
    # 4. Discover model folders (exclude MusicCaps/)
    # ------------------------------------------------------------------
    model_folders = discover_model_folders(extract_dir)
    if not model_folders:
        logger.error("No valid model folders found after extraction.")
        raise SystemExit(1)
    logger.info(
        "Found %d model folders: %s",
        len(model_folders),
        [m for m, _ in model_folders],
    )

    # ------------------------------------------------------------------
    # 5. Download MusicCaps CSV for caption matching
    # ------------------------------------------------------------------
    musiccaps_df = download_musiccaps_csv(staging_dir)
    caption_lookup = build_caption_lookup(musiccaps_df)
    logger.info("Caption lookup contains %d entries", len(caption_lookup))

    # ------------------------------------------------------------------
    # 6. Build processing tasks across all models
    # ------------------------------------------------------------------
    tasks: list[tuple] = []
    total_processed = 0

    # Distribute samples evenly across all discovered models
    per_model_limit = max(1, (args.num_samples + len(model_folders) - 1) // len(model_folders))

    for model_name, folder_path in model_folders:
        # Skip macOS resource-fork files (._<name>) which are not audio
        wav_files = sorted(f for f in folder_path.glob("*.wav") if not f.name.startswith("._"))
        model_count = 0
        for wav_file in wav_files:
            if model_count >= per_model_limit:
                break

            ytid = wav_file.stem
            out_filename = f"{model_name}_{ytid}.wav"

            if out_filename in existing_files:
                total_processed += 1
                model_count += 1
                continue

            caption = caption_lookup.get(ytid, "")
            meta = build_metadata_row(
                filename=out_filename,
                model=model_name,
                caption=caption,
            )
            tasks.append((wav_file, audio_dir / out_filename, meta))
            model_count += 1

    # ------------------------------------------------------------------
    # 7. Re-save WAVs as PCM_16 and collect metadata
    # ------------------------------------------------------------------
    metadata: list[dict] = []
    if csv_path.exists():
        metadata = pd.read_csv(csv_path).to_dict("records")

    if tasks:
        logger.info("Processing %d audio files...", len(tasks))
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(_resave_worker, t): t for t in tasks}
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Re-saving FakeMusicCaps audio",
            ):
                result = future.result()
                if result is not None:
                    metadata.append(result)
    else:
        logger.info("No new files to process (all already exist or num_samples reached).")

    # ------------------------------------------------------------------
    # 8. Write metadata.csv
    # ------------------------------------------------------------------
    df = pd.DataFrame(metadata)
    if not df.empty:
        df.to_csv(csv_path, index=False)
    logger.info("Saved %d total samples to %s", len(df), output_dir)
    logger.info("Metadata written to %s", csv_path)


if __name__ == "__main__":
    main()
