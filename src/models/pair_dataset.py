"""PyTorch Dataset for audio pair training."""
from __future__ import annotations

import hashlib
import logging
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.features.audio_features import extract_all_features, extract_features_batch, load_audio
from src.models.chunking import chunk_audio

logger = logging.getLogger(__name__)


def compute_feature_stats(cache_dir: str | Path) -> dict[str, 'np.ndarray']:
    """Compute per-dimension mean and std across all cached feature files.

    Scans all ``.npy`` files in *cache_dir*, computes running mean/std using
    Welford's online algorithm (numerically stable, single-pass).

    Returns ``{"mean": (feature_dim,), "std": (feature_dim,)}`` and saves the
    result as ``feature_stats.npz`` inside *cache_dir*.
    """
    cache_path = Path(cache_dir)
    npy_files = sorted(cache_path.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {cache_path}")

    # Welford's online algorithm for numerically stable mean/variance
    n = 0
    mean = None
    m2 = None

    for fpath in npy_files:
        try:
            arr = np.load(fpath)  # (n_chunks, feature_dim)
        except Exception:
            continue
        if arr.ndim != 2:
            continue
        for row in arr:
            n += 1
            if mean is None:
                mean = np.zeros_like(row, dtype=np.float64)
                m2 = np.zeros_like(row, dtype=np.float64)
            delta = row.astype(np.float64) - mean
            mean += delta / n
            delta2 = row.astype(np.float64) - mean
            m2 += delta * delta2

    if n < 2 or mean is None:
        raise ValueError(f"Not enough data to compute statistics (found {n} vectors)")

    std = np.sqrt(m2 / (n - 1)).astype(np.float32)
    mean_f32 = mean.astype(np.float32)

    stats_path = cache_path / "feature_stats.npz"
    np.savez(stats_path, mean=mean_f32, std=std)
    logger.info(
        "Computed feature stats from %d vectors (%d files) -> %s",
        n, len(npy_files), stats_path,
    )
    return {"mean": mean_f32, "std": std}


class AudioPairDataset(Dataset):
    """Dataset of audio pairs for training the siamese network.

    Expects a CSV with columns: track_a_path, track_b_path, label
    (1=similar/derivative, 0=unrelated).
    """

    def __init__(
        self,
        pairs_csv: str,
        feature_cache_dir: str | None = None,
        lmdb_path: str | None = None,
        window_sec: float = 10.0,
        stride_sec: float = 5.0,
        max_chunks: int = 12,
        sr: int = 16000,
        augmentor: object | None = None,
        n_augmentations: int = 0,
        feature_noise_std: float = 0.0,
        feature_dropout_p: float = 0.0,
        feature_stats_path: str | Path | None = None,
        training: bool = True,
    ) -> None:
        self.pairs = pd.read_csv(pairs_csv)
        self.cache_dir = Path(feature_cache_dir) if feature_cache_dir else None
        self.lmdb_path = lmdb_path
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.max_chunks = max_chunks
        self.sr = sr
        self.augmentor = augmentor
        self.n_augmentations = n_augmentations
        self.feature_noise_std = feature_noise_std
        self.feature_dropout_p = feature_dropout_p
        self.training = training

        # Feature standardization (per-dimension z-score)
        self._feat_mean: np.ndarray | None = None
        self._feat_std: np.ndarray | None = None
        if feature_stats_path is not None:
            stats = np.load(feature_stats_path)
            self._feat_mean = stats["mean"]
            self._feat_std = stats["std"]
            logger.info("Loaded feature stats from %s (dim=%d)", feature_stats_path, len(self._feat_mean))

        self._lmdb_env: Any = None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        required = {"track_a_path", "track_b_path", "label"}
        missing = required - set(self.pairs.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

    def _get_lmdb_env(self) -> Any:
        if self._lmdb_env is not None:
            return self._lmdb_env
        if self.lmdb_path is None:
            return None
        Path(self.lmdb_path).mkdir(parents=True, exist_ok=True)
        self._lmdb_env = lmdb.open(
            self.lmdb_path,
            map_size=1 << 40,
            readonly=False,
            lock=False,
            readahead=True,
            meminit=False,
        )
        return self._lmdb_env

    def __len__(self) -> int:
        return len(self.pairs)

    # ------------------------------------------------------------------
    # Caching helpers
    # ------------------------------------------------------------------

    def _cache_key(self, path: str, aug_variant: int = 0) -> str:
        raw = f"{path}|{self.window_sec}|{self.stride_sec}|{self.sr}"
        if aug_variant > 0:
            raw += f"|aug_{aug_variant}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _load_from_lmdb(self, key: str) -> np.ndarray | None:
        env = self._get_lmdb_env()
        if env is None:
            return None
        try:
            with env.begin(write=False) as txn:
                buf = txn.get(key.encode())
                if buf is None:
                    return None
                rows = int.from_bytes(buf[:8], "little")
                cols = int.from_bytes(buf[8:16], "little")
                arr = np.frombuffer(buf[16:], dtype=np.float32).reshape(rows, cols)
                return arr.copy()
        except Exception:
            return None

    def _save_to_lmdb(self, key: str, features: np.ndarray) -> None:
        env = self._get_lmdb_env()
        if env is None:
            return
        try:
            features = features.astype(np.float32)
            rows, cols = features.shape
            shape_bytes = rows.to_bytes(8, "little") + cols.to_bytes(8, "little")
            data = shape_bytes + features.tobytes()
            with env.begin(write=True) as txn:
                txn.put(key.encode(), data)
        except Exception as exc:
            logger.warning("Failed to write LMDB for key %s: %s", key, exc)

    def _load_cached(self, path: str, aug_variant: int = 0) -> np.ndarray | None:
        key = self._cache_key(path, aug_variant)
        result = self._load_from_lmdb(key)
        if result is not None:
            return result
        if self.cache_dir is None:
            return None
        cache_file = self.cache_dir / f"{key}.npy"
        if cache_file.exists():
            try:
                return np.load(cache_file)
            except Exception:
                return None
        return None

    def _save_cache(self, path: str, features: np.ndarray, aug_variant: int = 0) -> None:
        key = self._cache_key(path, aug_variant)
        env = self._get_lmdb_env()
        if env is not None:
            self._save_to_lmdb(key, features)
            return
        if self.cache_dir is None:
            return
        cache_file = self.cache_dir / f"{key}.npy"
        try:
            np.save(cache_file, features.astype(np.float32))
        except Exception as exc:
            logger.warning("Failed to cache features for %s: %s", path, exc)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, audio_path: str, aug_variant: int = 0) -> np.ndarray:
        """Load audio, chunk it, and extract per-chunk features.

        Returns array of shape `(num_chunks, feature_dim)`.
        """
        # Legacy live augmentation path (when no pre-augmented variants)
        if self.augmentor is not None and getattr(self.augmentor, "enabled", False) and aug_variant == 0:
            return self._extract_augmented(audio_path)

        # Try loading the requested variant from cache
        cached = self._load_cached(audio_path, aug_variant)
        if cached is not None:
            return cached
        # If a non-raw variant was requested but missing, try raw
        if aug_variant > 0:
            cached = self._load_cached(audio_path, 0)
        if cached is not None:
            return cached

        audio, sr = load_audio(audio_path, sr=self.sr)
        chunks = chunk_audio(audio, sr, self.window_sec, self.stride_sec)
        features = extract_features_batch(chunks, sr)
        self._save_cache(audio_path, features)
        return features

    def _extract_augmented(self, audio_path: str) -> np.ndarray:
        """Load audio, augment, chunk, extract — no caching."""
        audio, sr = load_audio(audio_path, sr=self.sr)
        waveform = torch.from_numpy(audio).float()
        waveform = self.augmentor(waveform)
        audio_aug = waveform.numpy() if isinstance(waveform, torch.Tensor) else waveform
        chunks = chunk_audio(audio_aug, sr, self.window_sec, self.stride_sec)
        return extract_features_batch(chunks, sr)

    def _apply_feature_perturbation(self, features: np.ndarray) -> np.ndarray:
        """Apply lightweight feature-space perturbations (Gaussian noise + dropout)."""
        if self.feature_noise_std > 0:
            features = features + np.random.normal(
                0, self.feature_noise_std, features.shape
            ).astype(features.dtype)
        if self.feature_dropout_p > 0:
            mask = np.random.random(features.shape) > self.feature_dropout_p
            features = features * mask.astype(features.dtype)
        return features

    def _pad_or_truncate(self, features: np.ndarray) -> np.ndarray:
        """Ensure exactly max_chunks rows via padding or truncation.

        During training a random start offset is used when the track has more
        chunks than max_chunks, preventing the model from over-fitting to
        the beginning of tracks (window-bias).  At eval/inference time the
        first max_chunks are always used for determinism.
        """
        n_chunks, feat_dim = features.shape
        if n_chunks >= self.max_chunks:
            if self.training:
                max_start = n_chunks - self.max_chunks
                start = random.randint(0, max_start)
            else:
                start = 0
            return features[start : start + self.max_chunks]
        padded = np.zeros((self.max_chunks, feat_dim), dtype=features.dtype)
        padded[:n_chunks] = features
        return padded

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> dict:
        """Return a single training pair.

        Keys
        ----
        features_a : Tensor of shape `(max_chunks, feature_dim)`
        features_b : Tensor of shape `(max_chunks, feature_dim)`
        label      : scalar float Tensor (0.0 or 1.0)
        """
        row = self.pairs.iloc[idx]

        # When pre-augmented variants exist, sample a random one per call
        if self.n_augmentations > 0:
            variant = random.randint(0, self.n_augmentations)
            features_a = self._extract_features(str(row["track_a_path"]), aug_variant=variant)
            features_b = self._extract_features(str(row["track_b_path"]), aug_variant=variant)
        else:
            features_a = self._extract_features(str(row["track_a_path"]))
            features_b = self._extract_features(str(row["track_b_path"]))

        features_a = self._pad_or_truncate(features_a)
        features_b = self._pad_or_truncate(features_b)

        # Per-dimension z-score standardization (applied before perturbation)
        if self._feat_mean is not None:
            features_a = (features_a - self._feat_mean) / (self._feat_std + 1e-8)
            features_b = (features_b - self._feat_mean) / (self._feat_std + 1e-8)

        # Lightweight feature-space perturbation on top of cached features
        if self.feature_noise_std > 0 or self.feature_dropout_p > 0:
            features_a = self._apply_feature_perturbation(features_a)
            features_b = self._apply_feature_perturbation(features_b)

        return {
            "features_a": torch.from_numpy(features_a).float(),
            "features_b": torch.from_numpy(features_b).float(),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
        }


def collate_pairs(batch: list[dict]) -> dict:
    """Custom collate that pads to the largest chunk count in the batch.

    Even though `AudioPairDataset` already pads to `max_chunks`, this
    collate function handles the general case where items may have different
    numbers of chunks (e.g. when caching is bypassed or a different Dataset
    implementation is used).
    """
    max_ca = max(b["features_a"].shape[0] for b in batch)
    max_cb = max(b["features_b"].shape[0] for b in batch)
    feat_dim = batch[0]["features_a"].shape[1]

    batched_a = torch.zeros(len(batch), max_ca, feat_dim)
    batched_b = torch.zeros(len(batch), max_cb, feat_dim)
    labels = torch.zeros(len(batch))

    for i, item in enumerate(batch):
        na = item["features_a"].shape[0]
        nb = item["features_b"].shape[0]
        batched_a[i, :na] = item["features_a"]
        batched_b[i, :nb] = item["features_b"]
        labels[i] = item["label"]

    return {
        "features_a": batched_a,
        "features_b": batched_b,
        "label": labels,
    }


# ---------------------------------------------------------------------------
# ProcessPoolExecutor-based feature pre-extraction
# ---------------------------------------------------------------------------


def _extract_and_cache_worker(
    audio_path: str,
    cache_dir: str,
    sr: int,
    window_sec: float,
    stride_sec: float,
) -> str:
    """Standalone worker function for ProcessPoolExecutor.

    Extracts features for a single audio file and saves to cache.
    Must be a module-level function (not a method) for pickling.
    """
    raw = f"{audio_path}|{window_sec}|{stride_sec}|{sr}"
    key = hashlib.sha256(raw.encode()).hexdigest()
    cache_file = Path(cache_dir) / f"{key}.npy"
    if cache_file.exists():
        return f"skip:{audio_path}"

    audio, loaded_sr = load_audio(audio_path, sr=sr)
    chunks = chunk_audio(audio, loaded_sr, window_sec, stride_sec)
    features = extract_features_batch(chunks, loaded_sr)
    np.save(cache_file, features.astype(np.float32))
    return f"done:{audio_path}"


def prefetch_features(
    pairs_csv: str,
    cache_dir: str,
    sr: int = 16000,
    window_sec: float = 10.0,
    stride_sec: float = 5.0,
    max_workers: int = 4,
) -> dict[str, int]:
    """Pre-extract features for all tracks in a pairs CSV using multiple processes.

    Uses ProcessPoolExecutor to parallelise CPU-bound feature extraction
    independently of the DataLoader's num_workers, which only parallelise
    I/O and collation.

    Returns:
        dict with counts: computed, skipped, failed.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pairs_csv)
    all_paths: set[str] = set()
    all_paths.update(df["track_a_path"].astype(str).tolist())
    all_paths.update(df["track_b_path"].astype(str).tolist())

    logger.info("Pre-fetching features for %d unique tracks (%d workers)", len(all_paths), max_workers)

    counts = {"computed": 0, "skipped": 0, "failed": 0}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _extract_and_cache_worker, path, cache_dir, sr, window_sec, stride_sec
            ): path
            for path in sorted(all_paths)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Pre-extracting"):
            path = futures[future]
            try:
                result = future.result()
                if result.startswith("skip:"):
                    counts["skipped"] += 1
                else:
                    counts["computed"] += 1
            except Exception as exc:
                logger.warning("Failed to extract features for %s: %s", path, exc)
                counts["failed"] += 1

    logger.info(
        "Pre-fetch done: %d computed, %d skipped, %d failed",
        counts["computed"],
        counts["skipped"],
        counts["failed"],
    )
    return counts
