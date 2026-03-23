"""Exploratory Data Analysis for AI Audio Similarity project.

Loads metadata from available datasets, prints distribution stats,
and generates spectrograms comparing real vs AI-generated audio.
Falls back to synthetic demo data if no real datasets are downloaded.

Usage:
    uv run python notebooks/01_eda.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FIGURES_DIR = PROJECT_ROOT / "notebooks" / "figures"
DATA_DIR = PROJECT_ROOT / "data"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", palette="muted")


# ── helpers ────────────────────────────────────────────────────

def load_dataset_metadata() -> dict[str, pd.DataFrame]:
    datasets: dict[str, pd.DataFrame] = {}

    sonics_csv = DATA_DIR / "sonics" / "metadata.csv"
    if sonics_csv.exists():
        logger.info("Loading SONICS metadata from %s", sonics_csv)
        datasets["sonics"] = pd.read_csv(sonics_csv)
    else:
        logger.warning("SONICS metadata not found at %s", sonics_csv)

    mippia_csv = DATA_DIR / "mippia" / "metadata.csv"
    if mippia_csv.exists():
        logger.info("Loading MIPPIA metadata from %s", mippia_csv)
        datasets["mippia"] = pd.read_csv(mippia_csv)
    else:
        logger.warning("MIPPIA metadata not found at %s", mippia_csv)

    fmc_csv = DATA_DIR / "fakemusiccaps" / "metadata.csv"
    if fmc_csv.exists():
        logger.info("Loading FakeMusicCaps metadata from %s", fmc_csv)
        datasets["fakemusiccaps"] = pd.read_csv(fmc_csv)
    else:
        logger.warning("FakeMusicCaps metadata not found at %s", fmc_csv)

    return datasets


def print_distribution_stats(datasets: dict[str, pd.DataFrame]) -> None:
    print("\n" + "=" * 60)
    print("  DATASET DISTRIBUTION STATISTICS")
    print("=" * 60)

    for name, df in datasets.items():
        print(f"\n--- {name.upper()} ({len(df)} rows) ---")
        print(f"  Columns: {list(df.columns)}")

        if "label" in df.columns:
            print("\n  Label distribution:")
            for label, count in df["label"].value_counts().items():
                print(f"    {label}: {count}")

        if "generator" in df.columns:
            print("\n  Generator distribution:")
            for gen, count in df["generator"].value_counts().items():
                print(f"    {gen}: {count}")
        elif "model" in df.columns:
            print("\n  Model distribution:")
            for model, count in df["model"].value_counts().items():
                print(f"    {model}: {count}")

        if "duration" in df.columns:
            dur = df["duration"]
            print("\n  Duration stats:")
            print(f"    min={dur.min():.1f}s  max={dur.max():.1f}s  "
                  f"mean={dur.mean():.1f}s  median={dur.median():.1f}s")


def plot_dataset_distributions(datasets: dict[str, pd.DataFrame]) -> None:
    if not datasets:
        return

    for name, df in datasets.items():
        if "label" in df.columns:
            fig, ax = plt.subplots(figsize=(8, 4))
            df["label"].value_counts().plot(kind="bar", ax=ax,
                                            color=sns.color_palette("muted"))
            ax.set_title(f"{name.upper()} — Track Count by Label")
            ax.set_ylabel("Count")
            ax.set_xlabel("Label")
            plt.tight_layout()
            fig.savefig(FIGURES_DIR / f"eda_{name}_label_dist.png", dpi=150)
            plt.close(fig)
            logger.info("Saved label distribution plot for %s", name)

        if "duration" in df.columns:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(df["duration"].dropna(), bins=30, edgecolor="black", alpha=0.7)
            ax.set_title(f"{name.upper()} — Duration Distribution")
            ax.set_xlabel("Duration (s)")
            ax.set_ylabel("Count")
            plt.tight_layout()
            fig.savefig(FIGURES_DIR / f"eda_{name}_duration_dist.png", dpi=150)
            plt.close(fig)
            logger.info("Saved duration distribution plot for %s", name)

        gen_col = ("generator" if "generator" in df.columns
                   else ("model" if "model" in df.columns else None))
        if gen_col:
            fig, ax = plt.subplots(figsize=(10, 5))
            df[gen_col].value_counts().plot(kind="barh", ax=ax,
                                            color=sns.color_palette("muted"))
            ax.set_title(f"{name.upper()} — Tracks by {gen_col.title()}")
            ax.set_xlabel("Count")
            plt.tight_layout()
            fig.savefig(FIGURES_DIR / f"eda_{name}_{gen_col}_dist.png", dpi=150)
            plt.close(fig)
            logger.info("Saved %s distribution plot for %s", gen_col, name)


def generate_synthetic_signals(
    sr: int = 16000, duration: float = 3.0,
) -> dict[str, np.ndarray]:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    real_signal = (
        0.5 * np.sin(2 * np.pi * 440 * t)
        + 0.3 * np.sin(2 * np.pi * 880 * t)
        + 0.15 * np.sin(2 * np.pi * 1320 * t)
        + 0.05 * np.random.randn(len(t))
    )
    envelope = np.exp(-0.3 * t) * (1 + 0.2 * np.sin(2 * np.pi * 5 * t))
    real_signal = (real_signal * envelope).astype(np.float32)

    ai_signal = (
        0.4 * np.sin(2 * np.pi * 440 * t)
        + 0.4 * np.sin(2 * np.pi * 880 * t)
        + 0.2 * np.random.randn(len(t))
    )
    ai_signal = ai_signal.astype(np.float32)

    return {"real": real_signal, "ai": ai_signal}


def plot_spectrograms(
    signals: dict[str, np.ndarray], sr: int = 16000, title_prefix: str = "",
) -> None:
    import librosa
    import librosa.display

    n_signals = len(signals)
    fig, axes = plt.subplots(n_signals, 2, figsize=(14, 4 * n_signals))
    if n_signals == 1:
        axes = axes.reshape(1, -1)

    for idx, (label, audio) in enumerate(signals.items()):
        time_axis = np.arange(len(audio)) / sr
        axes[idx, 0].plot(time_axis, audio, linewidth=0.5)
        axes[idx, 0].set_title(f"{title_prefix}{label} — Waveform")
        axes[idx, 0].set_xlabel("Time (s)")
        axes[idx, 0].set_ylabel("Amplitude")

        S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(
            S_db, sr=sr, x_axis="time", y_axis="mel", ax=axes[idx, 1],
        )
        axes[idx, 1].set_title(f"{title_prefix}{label} — Mel Spectrogram")
        fig.colorbar(img, ax=axes[idx, 1], format="%+2.0f dB")

    plt.tight_layout()
    safe_prefix = title_prefix.strip().lower().replace(" ", "_").replace(":", "")
    suffix = f"_{safe_prefix}" if safe_prefix else ""
    fname = f"eda_spectrograms{suffix}.png"
    fig.savefig(FIGURES_DIR / fname, dpi=150)
    plt.close(fig)
    logger.info("Saved spectrogram comparison: %s", fname)


def try_load_sample_audio(
    datasets: dict[str, pd.DataFrame],
) -> dict[str, np.ndarray] | None:
    import librosa

    if "sonics" not in datasets:
        return None

    df = datasets["sonics"]
    if "label" not in df.columns or "filename" not in df.columns:
        return None

    audio_dir = DATA_DIR / "sonics" / "audio"
    if not audio_dir.exists():
        return None

    signals: dict[str, np.ndarray] = {}
    for label_val in ["real", "ai"]:
        subset = df[df["label"] == label_val]
        if subset.empty:
            continue
        sample_file = audio_dir / subset.iloc[0]["filename"]
        if sample_file.exists():
            audio, _ = librosa.load(str(sample_file), sr=16000, mono=True)
            signals[label_val] = audio

    return signals if len(signals) >= 2 else None


# ── main ──────────────────────────────────────────────────────

def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  01 — EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    print("\n[1/4] Loading dataset metadata ...")
    datasets = load_dataset_metadata()

    if datasets:
        print(f"\n  Found {len(datasets)} dataset(s): {list(datasets.keys())}")
        print_distribution_stats(datasets)
        print("\n[2/4] Generating distribution plots ...")
        plot_dataset_distributions(datasets)
    else:
        print("\n  No dataset metadata found. Will use synthetic demo data.")

    print("\n[3/4] Generating spectrogram comparisons ...")
    real_audio = try_load_sample_audio(datasets) if datasets else None

    if real_audio:
        plot_spectrograms(real_audio, sr=16000, title_prefix="Dataset Sample: ")
        print("  Used real dataset samples for spectrograms.")
    else:
        print("  No audio files available. Generating synthetic demo signals ...")
        synth = generate_synthetic_signals()
        plot_spectrograms(synth, sr=16000, title_prefix="Synthetic Demo: ")
        print("  Used synthetic signals for spectrograms.")

    print("\n[4/4] Summary ...")
    if datasets:
        total_tracks = sum(len(df) for df in datasets.values())
        print(f"  Total metadata entries across all datasets: {total_tracks}")
        for name, df in datasets.items():
            print(f"    {name}: {len(df)} entries")
    else:
        print("  No real data available yet.")
        print("  Download datasets with:")
        print("    uv run python data/download_sonics.py")
        print("    uv run python data/download_mippia.py")
        print("    uv run python data/download_fakemusiccaps.py")

    print(f"\n  All figures saved to: {FIGURES_DIR}")
    print("\n" + "=" * 60)
    print("  EDA COMPLETE")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
