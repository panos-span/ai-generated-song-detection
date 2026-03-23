"""Pre-compute chunk-level features for all audio files referenced in a pairs CSV.

Saves features as .npy files so that training DataLoader becomes a simple
np.load() call.

Usage — CPU (default):
    uv run python data/precompute_features.py --pairs_csv data/pairs/train.csv --cache_dir data/feature_cache

Usage — GPU (5-50x faster on MFCC/Mel):
    uv run python data/precompute_features.py --pairs_csv data/pairs/train.csv --cache_dir data/feature_cache --use_gpu --batch_size 32

Usage — CPU, skip slow Parselmouth HNR:
    uv run python data/precompute_features.py --pairs_csv data/pairs/train.csv --cache_dir data/feature_cache --skip_hnr

Usage — Pre-generate augmented variants (hybrid augmentation):
    uv run python data/precompute_features.py --pairs_csv data/pairs/train.csv --cache_dir data/feature_cache --n_augmentations 5
"""

import argparse
import hashlib
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.audio_features import (
    extract_all_features,
    extract_features_batch_gpu,
    load_audio,
)
from src.features.augmentations import AudioAugmentor
from src.models.chunking import chunk_audio

logger = logging.getLogger(__name__)

# Per-worker globals initialised by _worker_init() — avoids re-loading state
# per file and safely propagates CLI flags into child processes.
_skip_hnr_global: bool = False
_augmentor_global: AudioAugmentor | None = None


def _worker_init(skip_hnr: bool, aug_prob: float = 0.3) -> None:
    """Initialise per-process globals (called by ProcessPoolExecutor initializer)."""
    global _skip_hnr_global, _augmentor_global
    _skip_hnr_global = skip_hnr
    _augmentor_global = AudioAugmentor(sr=16000, p=aug_prob, enabled=True)


def _cache_key(audio_path: str, window_sec: float, stride_sec: float, sr: int, aug_variant: int = 0) -> str:
    raw = f"{audio_path}|{window_sec}|{stride_sec}|{sr}"
    if aug_variant > 0:
        raw += f"|aug_{aug_variant}"
    return hashlib.sha256(raw.encode()).hexdigest()


def compute_and_save(
    audio_path: str,
    cache_dir: Path,
    sr: int,
    window_sec: float,
    stride_sec: float,
    aug_variant: int = 0,
) -> str:
    """Compute features for one audio file and save to cache (CPU worker).

    When *aug_variant* > 0, the global ``_augmentor_global`` is applied to the
    raw waveform before chunking, producing an augmented feature variant.
    """
    key = _cache_key(audio_path, window_sec, stride_sec, sr, aug_variant)
    cache_file = cache_dir / f"{key}.npy"
    if cache_file.exists():
        return f"skip:{audio_path}(v{aug_variant})"

    audio, sr_loaded = load_audio(audio_path, sr=sr)

    if aug_variant > 0 and _augmentor_global is not None:
        audio_t = torch.from_numpy(audio).float()
        audio_t = _augmentor_global(audio_t)
        audio = audio_t.numpy()

    chunks = chunk_audio(audio, sr_loaded, window_sec, stride_sec)
    chunk_feats = [
        extract_all_features(c, sr_loaded, skip_hnr=_skip_hnr_global)["combined"]
        for c in chunks
    ]
    np.save(cache_file, np.stack(chunk_feats).astype(np.float32))
    return f"done:{audio_path}(v{aug_variant})"


def _run_gpu_loop(
    all_paths: list[str],
    cache_dir: Path,
    sr: int,
    window_sec: float,
    stride_sec: float,
    batch_size: int,
    skip_hnr: bool,
    n_augmentations: int = 0,
    aug_prob: float = 0.3,
    train_paths: set[str] | None = None,
) -> tuple[int, int, int]:
    """Single-process GPU-accelerated feature extraction.

    Accumulates chunks from multiple audio files and flushes them to the GPU
    extractor in batches of ``batch_size``, amortising kernel-launch overhead.

    When *n_augmentations* > 0, each file is processed N+1 times: once raw
    (variant 0), plus N augmented variants (1..N).

    Returns (computed, skipped, failed).
    """
    from src.features.gpu_features import GPUFeatureExtractor

    gpu_extractor = GPUFeatureExtractor(sr=sr)
    augmentor = AudioAugmentor(sr=sr, p=aug_prob, enabled=True) if n_augmentations > 0 else None
    logger.info(f"GPU mode: device={gpu_extractor.device}, batch_size={batch_size}")

    skipped = computed = failed = 0

    # Build the full list of (path, variant) jobs.
    # Only training files get augmented variants.
    _train = train_paths or set()
    pending: list[tuple[str, int]] = []
    for path in all_paths:
        n_aug = n_augmentations if path in _train else 0
        for v in range(n_aug + 1):
            key = _cache_key(path, window_sec, stride_sec, sr, v)
            if (cache_dir / f"{key}.npy").exists():
                skipped += 1
            else:
                pending.append((path, v))

    logger.info(f"{skipped} file-variants already cached; {len(pending)} to compute")

    # Accumulated state for the current batch.
    batch_chunks: list[np.ndarray] = []
    batch_meta: list[tuple[str, int, int]] = []  # (audio_path, aug_variant, num_chunks)

    def _flush() -> None:
        """Run GPU inference on the current batch and persist results."""
        nonlocal computed, failed
        if not batch_chunks:
            return
        try:
            all_feats = extract_features_batch_gpu(
                batch_chunks, sr, gpu_extractor, skip_hnr=skip_hnr
            )
        except Exception as exc:
            for path, v, _ in batch_meta:
                logger.warning(f"GPU batch failed, skipping {path}(v{v}): {exc}")
                failed += 1
            return

        offset = 0
        for path, v, n_chunks in batch_meta:
            chunk_feats = all_feats[offset : offset + n_chunks]
            offset += n_chunks
            key = _cache_key(path, window_sec, stride_sec, sr, v)
            np.save(cache_dir / f"{key}.npy", chunk_feats.astype(np.float32))
            computed += 1

    for path, v in tqdm(pending, desc="Pre-computing features (GPU)"):
        try:
            audio, sr_loaded = load_audio(path, sr=sr)
            if v > 0 and augmentor is not None:
                audio_t = torch.from_numpy(audio).float()
                audio_t = augmentor(audio_t)
                audio = audio_t.numpy()
            chunks = chunk_audio(audio, sr_loaded, window_sec, stride_sec)
        except Exception as exc:
            logger.warning(f"Failed to load {path}(v{v}): {exc}")
            failed += 1
            continue

        batch_chunks.extend(chunks)
        batch_meta.append((path, v, len(chunks)))

        if len(batch_chunks) >= batch_size:
            _flush()
            batch_chunks.clear()
            batch_meta.clear()

    # Flush any remaining chunks.
    _flush()

    return computed, skipped, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute features for training pairs")
    parser.add_argument("--pairs_csv", required=True, nargs="+", help="One or more pair CSV files")
    parser.add_argument("--cache_dir", default="data/feature_cache")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--window_sec", type=float, default=10.0)
    parser.add_argument("--stride_sec", type=float, default=5.0)
    parser.add_argument(
        "--max_workers",
        type=int,
        default=min(os.cpu_count() or 4, 8),
        help="CPU worker processes (ignored when --use_gpu is set)",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        default=False,
        help="Use GPU-accelerated extraction (single process, large chunk batches)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of chunks to batch together for GPU inference (--use_gpu only)",
    )
    parser.add_argument(
        "--skip_hnr",
        action="store_true",
        default=False,
        help="Skip Parselmouth HNR computation (~50-200 ms per chunk), using 0.0 placeholder",
    )
    parser.add_argument(
        "--n_augmentations",
        type=int,
        default=0,
        help="Number of augmented variants per audio file (0=raw only, 5=recommended)",
    )
    parser.add_argument(
        "--aug_prob",
        type=float,
        default=0.3,
        help="Per-augmentation probability when generating augmented variants",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Collect paths from all CSVs; mark which ones come from a *train* CSV
    # so we can skip augmentation for val/test files (they never use
    # augmented variants during training).
    all_paths: set[str] = set()
    train_paths: set[str] = set()
    for csv_file in args.pairs_csv:
        df = pd.read_csv(csv_file)
        paths_in_csv = set(
            df["track_a_path"].astype(str).tolist()
            + df["track_b_path"].astype(str).tolist()
        )
        all_paths.update(paths_in_csv)
        if "train" in Path(csv_file).stem:
            train_paths.update(paths_in_csv)

    logger.info(f"Found {len(all_paths)} unique audio files to process")
    if train_paths:
        logger.info(f"  {len(train_paths)} from training CSVs (will be augmented)")
        logger.info(f"  {len(all_paths - train_paths)} from val/test CSVs (raw only)")
    if args.skip_hnr:
        logger.info("--skip_hnr active: Parselmouth HNR will be 0.0 (faster)")
    if args.n_augmentations > 0:
        logger.info(
            f"--n_augmentations={args.n_augmentations}: generating {args.n_augmentations} "
            f"augmented variants per training file (aug_prob={args.aug_prob})"
        )

    sorted_paths = sorted(all_paths)

    if args.use_gpu:
        computed, skipped, failed = _run_gpu_loop(
            sorted_paths,
            cache_dir,
            args.sr,
            args.window_sec,
            args.stride_sec,
            args.batch_size,
            args.skip_hnr,
            n_augmentations=args.n_augmentations,
            aug_prob=args.aug_prob,
            train_paths=train_paths,
        )
    else:
        skipped = computed = failed = 0

        # Build list of (path, variant) jobs: variant 0 is raw, 1..N are augmented.
        # Only training files get augmented variants; val/test get raw only.
        jobs: list[tuple[str, int]] = []
        for path in sorted_paths:
            n_aug = args.n_augmentations if path in train_paths else 0
            for v in range(n_aug + 1):
                jobs.append((path, v))

        with ProcessPoolExecutor(
            max_workers=args.max_workers,
            initializer=_worker_init,
            initargs=(args.skip_hnr, args.aug_prob),
        ) as executor:
            futures = {
                executor.submit(
                    compute_and_save,
                    path,
                    cache_dir,
                    args.sr,
                    args.window_sec,
                    args.stride_sec,
                    aug_variant,
                ): (path, aug_variant)
                for path, aug_variant in jobs
            }

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Pre-computing features"
            ):
                path, aug_variant = futures[future]
                try:
                    result = future.result()
                    if result.startswith("skip:"):
                        skipped += 1
                    else:
                        computed += 1
                except Exception as exc:
                    logger.warning(f"Failed to process {path}(v{aug_variant}): {exc}")
                    failed += 1

    logger.info(f"Done: {computed} computed, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
