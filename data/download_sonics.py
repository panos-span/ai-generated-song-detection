"""Download SONICS dataset: fake songs from Kaggle, real songs from YouTube.

Downloads metadata CSVs and individual fake MP3 files via the Kaggle Python
API (one file at a time -- avoids downloading the ~24 GB full-dataset zip).
Converts MP3 files to 16 kHz mono WAV PCM_16.  Real songs are downloaded
from YouTube using ``yt-dlp`` with the ``youtube_id`` column.

Usage::

    python data/download_sonics.py --num_samples 1000 --output_dir data/sonics
"""
from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import librosa
import pandas as pd
import soundfile as sf
import yt_dlp
from tqdm import tqdm

logger = logging.getLogger(__name__)

TARGET_SR = 16000


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download SONICS dataset (Kaggle fake songs + YouTube real songs)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Total number of tracks to process (fake + real combined)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/sonics",
        help="Output directory (default: data/sonics)",
    )
    parser.add_argument(
        "--kaggle_dataset",
        type=str,
        default="awsaf49/sonics-dataset",
        help="Kaggle dataset identifier (default: awsaf49/sonics-dataset)",
    )
    return parser.parse_args()


# ======================================================================
# Metadata helpers
# ======================================================================

def build_metadata_row_fake(
    filename: str,
    source: str,
    label: str,
    split: str,
) -> dict:
    """Build a metadata row for a fake song.

    All fake label variants ('full fake', 'half fake', 'mostly fake') are
    mapped to ``label='ai'``.  The ``source`` column (suno / udio) becomes
    ``generator``.
    """
    return {
        "filename": filename,
        "label": "ai",
        "generator": source,
        "split": split,
    }


def build_metadata_row_real(
    filename: str,
    split: str,
) -> dict:
    """Build a metadata row for a real song."""
    return {
        "filename": filename,
        "label": "real",
        "generator": "real",
        "split": split,
    }


def get_existing_filenames(csv_path: Path) -> set[str]:
    """Return filenames already recorded in *csv_path*, or an empty set."""
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    return set(df["filename"].tolist())


# ======================================================================
# Kaggle helpers (individual file downloads --- avoids the ~24 GB zip)
# ======================================================================

def _patched_kaggle_api():
    """Return an authenticated KaggleApi with SSL verification disabled.

    Corporate proxies inject custom root certificates that fail verification;
    disabling it is the same workaround used for the FakeMusicCaps download.
    """
    try:
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context

        import requests as _requests
        _orig = _requests.Session.__init__

        def _patched(self, *a, **kw):
            _orig(self, *a, **kw)
            self.verify = False

        _requests.Session.__init__ = _patched
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        return api
    except ImportError:
        logger.error(
            "kaggle package not installed. Install with `pip install kaggle` "
            "and set KAGGLE_USERNAME and KAGGLE_KEY env vars."
        )
        sys.exit(1)
    except Exception as exc:
        msg = str(exc)
        if "401" in msg or "unauthorized" in msg.lower():
            logger.error(
                "Kaggle authentication failed (401). "
                "Ensure KAGGLE_USERNAME and KAGGLE_KEY are set."
            )
        else:
            logger.error("Kaggle auth failed: %s", exc)
        sys.exit(1)


def ensure_kaggle_csvs(api, dataset: str, staging_dir: Path) -> tuple[Path, Path]:
    """Download fake_songs.csv and real_songs.csv individually if not present."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    fake_csv = staging_dir / "fake_songs.csv"
    real_csv = staging_dir / "real_songs.csv"

    if not fake_csv.exists():
        logger.info("Downloading fake_songs.csv from Kaggle...")
        api.dataset_download_file(dataset, "fake_songs.csv", str(staging_dir))
    if not fake_csv.exists():
        logger.error("fake_songs.csv not found after download")
        sys.exit(1)
    logger.info("fake_songs.csv ready (%d bytes)", fake_csv.stat().st_size)

    if not real_csv.exists():
        logger.info("Downloading real_songs.csv from Kaggle...")
        api.dataset_download_file(dataset, "real_songs.csv", str(staging_dir))
    if not real_csv.exists():
        logger.error("real_songs.csv not found after download")
        sys.exit(1)
    logger.info("real_songs.csv ready (%d bytes)", real_csv.stat().st_size)

    return fake_csv, real_csv


def ensure_fake_mp3(api, dataset: str, filename: str, mp3_dir: Path) -> Path:
    """Download a single fake MP3 from Kaggle if not already present.

    The filename in fake_songs.csv has no extension; the Kaggle file lives at
    ``fake_songs/{filename}.mp3`` within the dataset.
    """
    mp3_path = mp3_dir / f"{filename}.mp3"
    if mp3_path.exists():
        return mp3_path
    mp3_dir.mkdir(parents=True, exist_ok=True)
    try:
        api.dataset_download_file(dataset, f"fake_songs/{filename}.mp3", str(mp3_dir))
    except Exception as exc:
        logger.warning("Kaggle download failed for %s: %s", filename, exc)
    return mp3_path


# ======================================================================
# Audio conversion
# ======================================================================

def convert_mp3_to_wav(mp3_path: Path, wav_path: Path) -> bool:
    """Load an MP3, resample to 16 kHz mono, and save as WAV PCM_16."""
    try:
        y, sr = librosa.load(str(mp3_path), sr=TARGET_SR, mono=True)
        sf.write(str(wav_path), y, TARGET_SR, subtype="PCM_16")
        return True
    except Exception as e:
        logger.warning("Error converting %s: %s", mp3_path.name, e)
        return False


def _convert_worker(args: tuple) -> dict | None:
    mp3_path, wav_path, metadata_row = args
    ok = convert_mp3_to_wav(mp3_path, wav_path)
    return metadata_row if ok else None


# ======================================================================
# YouTube download (real songs)
# ======================================================================

def download_real_song(youtube_id: str, output_path: Path) -> bool:
    """Download a single YouTube video as 16 kHz mono WAV via yt-dlp.

    Returns True on success, False on failure (logged, not raised).
    """
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(output_path.with_suffix("")),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        "postprocessor_args": ["-ar", "16000", "-ac", "1"],
        "quiet": True,
        "no_warnings": True,
        "nocheckcertificate": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={youtube_id}"])
    except Exception as exc:
        logger.error("yt-dlp failed for youtube_id=%s: %s", youtube_id, exc)
        return False
    return output_path.exists()


def _download_real_worker(args: tuple) -> dict | None:
    youtube_id, wav_path, metadata_row = args
    ok = download_real_song(youtube_id, wav_path)
    return metadata_row if ok else None


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    output_dir = Path(args.output_dir)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metadata.csv"

    existing_files = get_existing_filenames(csv_path)
    if existing_files:
        logger.info("Found %d already-processed files, will skip them", len(existing_files))

    # ------------------------------------------------------------------
    # 1. Authenticate Kaggle + ensure CSVs are present
    # ------------------------------------------------------------------
    staging_dir = output_dir / "_staging"
    api = _patched_kaggle_api()
    logger.info("Kaggle authenticated")
    fake_csv, real_csv = ensure_kaggle_csvs(api, args.kaggle_dataset, staging_dir)

    fake_df = pd.read_csv(fake_csv, low_memory=False)
    real_df = pd.read_csv(real_csv, low_memory=False)
    logger.info("Loaded %d fake songs, %d real songs from Kaggle CSVs", len(fake_df), len(real_df))

    # ------------------------------------------------------------------
    # 2. Determine how many fake / real to process
    # ------------------------------------------------------------------
    total_available = len(fake_df) + len(real_df)
    num_total = min(args.num_samples, total_available)

    # Split proportionally
    fake_ratio = len(fake_df) / total_available if total_available else 0.5
    num_fake = int(num_total * fake_ratio)
    num_real = num_total - num_fake

    # Ensure at least 1 of each when possible
    if num_fake == 0 and len(fake_df) > 0 and num_total >= 2:
        num_fake = 1
        num_real = num_total - 1
    if num_real == 0 and len(real_df) > 0 and num_total >= 2:
        num_real = 1
        num_fake = num_total - 1

    fake_subset = fake_df.head(num_fake)
    real_subset = real_df.head(num_real)

    # ------------------------------------------------------------------
    # 3. Download fake MP3s individually from Kaggle, then convert to WAV
    # ------------------------------------------------------------------
    mp3_dir = staging_dir / "fake_songs"
    mp3_dir.mkdir(parents=True, exist_ok=True)

    fake_tasks: list[tuple] = []
    logger.info("Downloading %d fake MP3s from Kaggle (individually)...", num_fake)
    for _, row in tqdm(fake_subset.iterrows(), total=len(fake_subset), desc="Fetching fake MP3s"):
        mp3_name = str(row["filename"])
        wav_name = mp3_name + ".wav"
        if wav_name in existing_files:
            continue
        mp3_path = ensure_fake_mp3(api, args.kaggle_dataset, mp3_name, mp3_dir)
        if not mp3_path.exists():
            logger.warning("MP3 not available after download: %s", mp3_name)
            continue
        meta = build_metadata_row_fake(
            filename=wav_name,
            source=str(row.get("source", "unknown")),
            label=str(row.get("label", "unknown")),
            split=str(row.get("split", "train")),
        )
        fake_tasks.append((mp3_path, audio_dir / wav_name, meta))

    metadata: list[dict] = []
    if csv_path.exists():
        metadata = pd.read_csv(csv_path).to_dict("records")

    if fake_tasks:
        logger.info("Converting %d fake songs (MP3 → WAV)...", len(fake_tasks))
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(_convert_worker, t): t for t in fake_tasks}
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Converting fake songs"
            ):
                result = future.result()
                if result is not None:
                    metadata.append(result)

    # ------------------------------------------------------------------
    # 4. Process real songs (YouTube → WAV)
    # ------------------------------------------------------------------
    real_tasks: list[tuple] = []
    for _, row in real_subset.iterrows():
        youtube_id = str(row.get("youtube_id", "")).strip()
        if not youtube_id:
            continue
        wav_name = f"real_{youtube_id}.wav"
        if wav_name in existing_files:
            continue
        meta = build_metadata_row_real(
            filename=wav_name,
            split=str(row.get("split", "train")),
        )
        real_tasks.append((youtube_id, audio_dir / wav_name, meta))

    if real_tasks:
        logger.info("Downloading %d real songs from YouTube...", len(real_tasks))
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_download_real_worker, t): t for t in real_tasks}
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Downloading real songs"
            ):
                result = future.result()
                if result is not None:
                    metadata.append(result)

    # ------------------------------------------------------------------
    # 5. Write metadata
    # ------------------------------------------------------------------
    df = pd.DataFrame(metadata)
    if not df.empty:
        df.to_csv(csv_path, index=False)
    logger.info("Saved %d total samples to %s", len(df), output_dir)
    logger.info("Metadata written to %s", csv_path)


if __name__ == "__main__":
    main()
