import argparse
import logging
import subprocess
from pathlib import Path

import pandas as pd
import yt_dlp
from tqdm import tqdm

logger = logging.getLogger(__name__)

REPO_URL = "https://github.com/Mippia/smp_dataset.git"
REPO_DIR = "smp_dataset"
CSV_FILENAME = "Final_dataset_pairs.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download MIPPIA SMP dataset from GitHub")
    parser.add_argument("--output_dir", type=str, default="data/mippia", help="Output directory")
    return parser.parse_args()


def clone_repo(output_dir: Path) -> Path:
    repo_path = output_dir / REPO_DIR
    if repo_path.exists():
        logger.info(f"Repository already cloned at {repo_path}")
        return repo_path

    logger.info(f"Cloning {REPO_URL}...")
    subprocess.run(
        ["git", "clone", REPO_URL, str(repo_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    logger.info("Clone complete")
    return repo_path


def download_audio(url: str, output_path: Path) -> bool:
    if output_path.exists():
        return True
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
            ydl.download([url])
    except Exception as exc:
        logger.error(f"yt-dlp failed for {url}: {exc}")
        return False
    return output_path.exists()


def sanitize_filename(title: str) -> str:
    return "".join(c if c.isalnum() or c in " -_" else "_" for c in title).strip()[:100]


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    output_dir = Path(args.output_dir)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    existing_pairs: set[int] = set()
    meta_csv_path = output_dir / "metadata.csv"
    if meta_csv_path.exists():
        existing_df = pd.read_csv(meta_csv_path)
        completed = existing_df[(existing_df["track_a"] != "") & (existing_df["track_b"] != "")]
        existing_pairs = set(completed["pair_id"].tolist())
        logger.info(f"Found {len(existing_pairs)} already-completed pairs, will skip them")

    repo_path = clone_repo(output_dir)
    csv_path = repo_path / CSV_FILENAME
    if not csv_path.exists():
        logger.error(f"CSV not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows from CSV")

    unique_pairs = df.drop_duplicates(
        subset=["pair_number", "ori_title", "comp_title", "ori_link", "comp_link"]
    )
    logger.info(f"Found {len(unique_pairs)} unique pairs")

    metadata: list[dict] = []
    if meta_csv_path.exists():
        metadata = pd.read_csv(meta_csv_path).to_dict("records")

    for _, row in tqdm(unique_pairs.iterrows(), total=len(unique_pairs), desc="Downloading MIPPIA"):
        pair_id = int(row["pair_number"])
        if pair_id in existing_pairs:
            continue

        relation = row.get("relation", "unknown")

        ori_title = sanitize_filename(str(row["ori_title"]))
        comp_title = sanitize_filename(str(row["comp_title"]))

        track_a_file = f"pair{pair_id:03d}_a_{ori_title[:50]}.wav"
        track_b_file = f"pair{pair_id:03d}_b_{comp_title[:50]}.wav"

        track_a_ok = False
        track_b_ok = False

        try:
            track_a_ok = download_audio(row["ori_link"], audio_dir / track_a_file)
        except Exception as e:
            logger.error(f"Failed to download track A for pair {pair_id}: {e}")

        try:
            track_b_ok = download_audio(row["comp_link"], audio_dir / track_b_file)
        except Exception as e:
            logger.error(f"Failed to download track B for pair {pair_id}: {e}")

        metadata.append(
            {
                "pair_id": pair_id,
                "track_a": track_a_file if track_a_ok else "",
                "track_b": track_b_file if track_b_ok else "",
                "similarity_label": relation,
            }
        )

    meta_df = pd.DataFrame(metadata)
    meta_df.to_csv(meta_csv_path, index=False)
    logger.info(f"Saved metadata for {len(meta_df)} pairs to {meta_csv_path}")


if __name__ == "__main__":
    main()
