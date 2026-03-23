"""Standalone script to build training pairs from downloaded datasets.

Reads metadata CSVs from data/sonics, data/mippia, data/fakemusiccaps and
constructs positive, negative, and hard-negative pairs for training,
validation, and test splits.

Usage::

    python -m src.models.construct_pairs --data_dir data --output_dir data/pairs
"""
from __future__ import annotations

import argparse
import logging
import random
from itertools import combinations
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


# ======================================================================
# Dataset loaders
# ======================================================================

def _load_sonics(data_dir: Path) -> pd.DataFrame | None:
    csv_path = data_dir / "sonics" / "metadata.csv"
    if not csv_path.exists():
        logger.warning("SONICS metadata not found at %s", csv_path)
        return None
    df = pd.read_csv(csv_path)
    audio_dir = data_dir / "sonics" / "audio"
    df["full_path"] = df["filename"].apply(lambda f: str(audio_dir / f))
    df = df[df["full_path"].apply(lambda p: Path(p).exists())]
    logger.info("Loaded %d SONICS tracks", len(df))
    return df


def _load_mippia(data_dir: Path) -> pd.DataFrame | None:
    csv_path = data_dir / "mippia" / "metadata.csv"
    if not csv_path.exists():
        logger.warning("MIPPIA metadata not found at %s", csv_path)
        return None
    df = pd.read_csv(csv_path)
    audio_dir = data_dir / "mippia" / "audio"
    df["track_a_full"] = df["track_a"].apply(
        lambda f: str(audio_dir / f) if pd.notna(f) and f else ""
    )
    df["track_b_full"] = df["track_b"].apply(
        lambda f: str(audio_dir / f) if pd.notna(f) and f else ""
    )
    df = df[
        (df["track_a_full"] != "")
        & (df["track_b_full"] != "")
        & df["track_a_full"].apply(lambda p: Path(p).exists())
        & df["track_b_full"].apply(lambda p: Path(p).exists())
    ]
    logger.info("Loaded %d MIPPIA pairs with audio", len(df))
    return df


def _load_fakemusiccaps(data_dir: Path) -> pd.DataFrame | None:
    csv_path = data_dir / "fakemusiccaps" / "metadata.csv"
    if not csv_path.exists():
        logger.warning("FakeMusicCaps metadata not found at %s", csv_path)
        return None
    df = pd.read_csv(csv_path)
    audio_dir = data_dir / "fakemusiccaps" / "audio"
    df["full_path"] = df["filename"].apply(lambda f: str(audio_dir / f))
    df = df[df["full_path"].apply(lambda p: Path(p).exists())]
    logger.info("Loaded %d FakeMusicCaps tracks", len(df))
    return df


# ======================================================================
# Pair builders
# ======================================================================

def _sample_pairs(
    list_a: list[str],
    list_b: list[str],
    n: int,
    rng: random.Random,
) -> list[tuple[str, str]]:
    """Sample up to *n* unique (a, b) pairs from two pools."""
    pairs: set[tuple[str, str]] = set()
    max_attempts = n * 5
    for _ in range(max_attempts):
        if len(pairs) >= n:
            break
        a = rng.choice(list_a)
        b = rng.choice(list_b)
        if a != b:
            pairs.add((a, b))
    return list(pairs)


def _build_sonics_pairs(df: pd.DataFrame, rng: random.Random) -> list[dict]:
    """Build positive, negative, and hard-negative pairs from SONICS."""
    real = df[df["label"] == "real"]["full_path"].tolist()
    ai = df[df["label"] == "ai"]["full_path"].tolist()
    if not real or not ai:
        logger.warning("SONICS: need both real and AI tracks for pair construction")
        return []

    pairs: list[dict] = []
    n_pos = min(len(real) * len(ai), max(len(real), len(ai)))

    for a, b in _sample_pairs(real, ai, n_pos, rng):
        pairs.append(
            {"track_a_path": a, "track_b_path": b, "label": 1, "pair_type": "positive"}
        )

    n_neg = len(pairs)
    for a, b in _sample_pairs(real, real, n_neg, rng):
        pairs.append(
            {"track_a_path": a, "track_b_path": b, "label": 0, "pair_type": "negative"}
        )

    generators = df[df["label"] == "ai"].groupby("generator")["full_path"].apply(list).to_dict()
    hard_pairs: list[dict] = []
    for gen, tracks in generators.items():
        if len(tracks) < 2:
            continue
        sampled = list(combinations(tracks[:50], 2))
        rng.shuffle(sampled)
        for a, b in sampled[: max(10, len(sampled) // 4)]:
            hard_pairs.append(
                {"track_a_path": a, "track_b_path": b, "label": 0, "pair_type": "hard_negative"}
            )
    pairs.extend(hard_pairs)
    logger.info("SONICS: %d pairs (pos=%d, neg=%d, hard_neg=%d)",
                len(pairs),
                sum(1 for p in pairs if p["pair_type"] == "positive"),
                sum(1 for p in pairs if p["pair_type"] == "negative"),
                sum(1 for p in pairs if p["pair_type"] == "hard_negative"))
    return pairs


def _build_mippia_pairs(df: pd.DataFrame, rng: random.Random) -> list[dict]:
    """Build positive pairs from MIPPIA explicit similarity annotations."""
    pairs: list[dict] = []
    tracks_a = df["track_a_full"].tolist()
    tracks_b = df["track_b_full"].tolist()

    for a, b in zip(tracks_a, tracks_b):
        pairs.append(
            {"track_a_path": a, "track_b_path": b, "label": 1, "pair_type": "positive"}
        )

    all_tracks = list(set(tracks_a + tracks_b))
    n_neg = len(pairs)
    for a, b in _sample_pairs(all_tracks, all_tracks, n_neg, rng):
        if not any(
            (p["track_a_path"] == a and p["track_b_path"] == b)
            or (p["track_a_path"] == b and p["track_b_path"] == a)
            for p in pairs
            if p["pair_type"] == "positive"
        ):
            pairs.append(
                {"track_a_path": a, "track_b_path": b, "label": 0, "pair_type": "negative"}
            )

    logger.info("MIPPIA: %d pairs", len(pairs))
    return pairs


def _build_cross_dataset_negatives(
    sonics_df: pd.DataFrame | None,
    fmc_df: pd.DataFrame | None,
    rng: random.Random,
    max_pairs: int = 500,
) -> list[dict]:
    """Build negative pairs across SONICS and FakeMusicCaps."""
    pairs: list[dict] = []
    if sonics_df is None or fmc_df is None:
        return pairs

    real = sonics_df[sonics_df["label"] == "real"]["full_path"].tolist()
    fmc = fmc_df["full_path"].tolist()
    if not real or not fmc:
        return pairs

    for a, b in _sample_pairs(real, fmc, max_pairs, rng):
        pairs.append(
            {"track_a_path": a, "track_b_path": b, "label": 0, "pair_type": "negative"}
        )
    logger.info("Cross-dataset negatives: %d pairs", len(pairs))
    return pairs


def _build_fmc_hard_negatives(df: pd.DataFrame, rng: random.Random) -> list[dict]:
    """Hard negatives from FakeMusicCaps: same model, different tracks."""
    pairs: list[dict] = []
    grouped = df.groupby("model")["full_path"].apply(list).to_dict()
    for model, tracks in grouped.items():
        if len(tracks) < 2:
            continue
        combos = list(combinations(tracks[:40], 2))
        rng.shuffle(combos)
        for a, b in combos[: max(10, len(combos) // 4)]:
            pairs.append(
                {"track_a_path": a, "track_b_path": b, "label": 0, "pair_type": "hard_negative"}
            )
    if pairs:
        logger.info("FakeMusicCaps hard negatives: %d pairs", len(pairs))
    return pairs


def _build_fmc_intra_positives(df: pd.DataFrame, rng: random.Random) -> list[dict]:
    """Positive pairs from FakeMusicCaps: different models, same YouTube source ID.

    Two AI tracks generated from the same source clip (same YouTube ID, different
    models) share the same underlying musical content and are treated as similar
    (label=1).  This allows positive-pair construction from FMC alone, without
    needing external real-audio counterparts.
    """
    df = df.copy()
    # Filenames are like "MusicGen_medium_<ytid>.wav" and "stable_audio_open_<ytid>.wav".
    # Split on the last underscore to isolate the YouTube ID.
    df["ytid"] = (
        df["filename"].str.rsplit("_", n=1).str[-1].str.replace(".wav", "", regex=False)
    )

    pairs: list[dict] = []
    for ytid, group in df.groupby("ytid"):
        if len(group) < 2:
            continue
        tracks = group["full_path"].tolist()
        combos = list(combinations(tracks, 2))
        rng.shuffle(combos)
        # Cap at 5 pairs per YouTube ID to avoid over-representing any single source
        for a, b in combos[:5]:
            pairs.append(
                {"track_a_path": a, "track_b_path": b, "label": 1, "pair_type": "fmc_positive"}
            )

    logger.info("FakeMusicCaps intra-positives: %d pairs", len(pairs))
    return pairs


# ======================================================================
# Main construction logic
# ======================================================================

def construct_all_pairs(
    data_dir: str,
    output_dir: str,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
) -> None:
    """Construct training pair CSVs from all available datasets.

    Parameters
    ----------
    data_dir:
        Root data directory containing `sonics/`, `mippia/`, `fakemusiccaps/`.
    output_dir:
        Directory where `train_pairs.csv`, `val_pairs.csv`, `test_pairs.csv` are written.
    val_split:
        Fraction of pairs for validation.
    test_split:
        Fraction of pairs for testing.
    seed:
        Random seed for reproducibility.
    """
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    sonics_df = _load_sonics(data_path)
    mippia_df = _load_mippia(data_path)
    fmc_df = _load_fakemusiccaps(data_path)

    all_pairs: list[dict] = []

    if sonics_df is not None and not sonics_df.empty:
        all_pairs.extend(_build_sonics_pairs(sonics_df, rng))

    if mippia_df is not None and not mippia_df.empty:
        all_pairs.extend(_build_mippia_pairs(mippia_df, rng))

    all_pairs.extend(_build_cross_dataset_negatives(sonics_df, fmc_df, rng))

    if fmc_df is not None and not fmc_df.empty:
        all_pairs.extend(_build_fmc_hard_negatives(fmc_df, rng))
        all_pairs.extend(_build_fmc_intra_positives(fmc_df, rng))

    if not all_pairs:
        logger.error("No pairs constructed. Ensure datasets are downloaded.")
        return

    df = pd.DataFrame(all_pairs)
    rng.shuffle(all_pairs)

    # --- Pair balance validation ---
    n_pos = int((df["label"] == 1).sum())
    n_neg = int((df["label"] == 0).sum())
    ratio = n_pos / max(n_neg, 1)
    logger.info(
        "Pair balance: positive=%d, negative=%d, ratio=%.2f",
        n_pos, n_neg, ratio,
    )
    if ratio < 0.3 or ratio > 3.0:
        logger.warning(
            "Pair balance is skewed (ratio=%.2f). Consider adjusting pair "
            "construction to improve positive/negative balance.",
            ratio,
        )
    pair_types = df["pair_type"].value_counts().to_dict()
    logger.info("Pair types: %s", pair_types)

    test_frac = test_split
    val_frac = val_split / (1.0 - test_split)

    train_val, test_df = train_test_split(
        df, test_size=test_frac, random_state=seed, stratify=df["label"],
    )
    train_df, val_df = train_test_split(
        train_val, test_size=val_frac, random_state=seed, stratify=train_val["label"],
    )

    train_df.to_csv(out_path / "train_pairs.csv", index=False)
    val_df.to_csv(out_path / "val_pairs.csv", index=False)
    test_df.to_csv(out_path / "test_pairs.csv", index=False)

    logger.info(
        "Pairs written -> train=%d, val=%d, test=%d  (%s)",
        len(train_df), len(val_df), len(test_df), out_path,
    )


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build training pairs from downloaded audio datasets",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Root directory containing sonics/, mippia/, fakemusiccaps/ (default: data)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/pairs",
        help="Output directory for train/val/test CSVs (default: data/pairs)",
    )
    parser.add_argument("--val_split", type=float, default=0.15, help="Validation fraction")
    parser.add_argument("--test_split", type=float, default=0.15, help="Test fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    construct_all_pairs(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
