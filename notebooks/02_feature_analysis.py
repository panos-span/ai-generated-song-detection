"""Feature Engineering Analysis for AI Audio Similarity project.

Extracts Tier 1 and Tier 2 audio features from sample tracks (or synthetic
audio), prints statistics, and identifies which features best differentiate
real-like vs AI-like signals.

Usage:
    uv run python notebooks/02_feature_analysis.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.audio_features import (
    extract_all_features,
    extract_tier1_features,
    extract_tier2_features,
)

FIGURES_DIR = PROJECT_ROOT / "notebooks" / "figures"
DATA_DIR = PROJECT_ROOT / "data"
SR = 16000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", palette="muted")


# ── synthetic signal generators ────────────────────────────────

def _make_real_like(duration: float = 3.0) -> np.ndarray:
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    sig = (
        0.5 * np.sin(2 * np.pi * 440 * t)
        + 0.3 * np.sin(2 * np.pi * 880 * t)
        + 0.15 * np.sin(2 * np.pi * 1320 * t)
        + 0.04 * np.random.randn(len(t))
    )
    envelope = np.exp(-0.2 * t) * (1 + 0.3 * np.sin(2 * np.pi * 4 * t))
    return (sig * envelope).astype(np.float32)


def _make_ai_like(duration: float = 3.0) -> np.ndarray:
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    sig = (
        0.4 * np.sin(2 * np.pi * 440 * t)
        + 0.4 * np.sin(2 * np.pi * 880 * t)
        + 0.25 * np.random.randn(len(t))
    )
    return sig.astype(np.float32)


def generate_sample_pool(n_per_class: int = 5) -> dict[str, list[np.ndarray]]:
    np.random.seed(42)
    pool: dict[str, list[np.ndarray]] = {"real": [], "ai": []}
    for i in range(n_per_class):
        dur = 2.0 + i * 0.5
        pool["real"].append(_make_real_like(dur))
        pool["ai"].append(_make_ai_like(dur))
    return pool


def try_load_audio_pool() -> dict[str, list[np.ndarray]] | None:
    import librosa

    audio_dir = DATA_DIR / "sonics" / "audio"
    meta_csv = DATA_DIR / "sonics" / "metadata.csv"
    if not meta_csv.exists() or not audio_dir.exists():
        return None

    import pandas as pd
    df = pd.read_csv(meta_csv)
    if "label" not in df.columns or "filename" not in df.columns:
        return None

    pool: dict[str, list[np.ndarray]] = {"real": [], "ai": []}
    for label_val in ["real", "ai"]:
        subset = df[df["label"] == label_val].head(5)
        for _, row in subset.iterrows():
            fpath = audio_dir / row["filename"]
            if fpath.exists():
                audio, _ = librosa.load(str(fpath), sr=SR, mono=True)
                pool[label_val].append(audio)

    if len(pool["real"]) < 2 or len(pool["ai"]) < 2:
        return None
    return pool


# ── analysis ───────────────────────────────────────────────────

def extract_feature_matrix(
    pool: dict[str, list[np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    results: dict[str, dict[str, np.ndarray]] = {}
    for label, signals in pool.items():
        tier1_list, tier2_list, combined_list = [], [], []
        for audio in signals:
            feats = extract_all_features(audio, SR)
            tier1_list.append(feats["tier1"])
            tier2_list.append(feats["tier2"])
            combined_list.append(feats["combined"])
        results[label] = {
            "tier1": np.array(tier1_list),
            "tier2": np.array(tier2_list),
            "combined": np.array(combined_list),
        }
    return results


def print_feature_stats(feat_matrices: dict[str, dict[str, np.ndarray]]) -> None:
    print("\n" + "=" * 60)
    print("  FEATURE VECTOR STATISTICS")
    print("=" * 60)
    for label, tiers in feat_matrices.items():
        print(f"\n--- {label.upper()} ---")
        for tier_name, matrix in tiers.items():
            print(f"  {tier_name}: shape={matrix.shape}  "
                  f"mean={matrix.mean():.4f}  std={matrix.std():.4f}  "
                  f"min={matrix.min():.4f}  max={matrix.max():.4f}")


def compute_feature_importance(
    feat_matrices: dict[str, dict[str, np.ndarray]],
) -> dict[str, list[tuple[int, float, float]]]:
    importance: dict[str, list[tuple[int, float, float]]] = {}

    for tier_name in ["tier1", "tier2", "combined"]:
        real_mat = feat_matrices["real"][tier_name]
        ai_mat = feat_matrices["ai"][tier_name]
        n_features = real_mat.shape[1]
        results: list[tuple[int, float, float]] = []

        for i in range(n_features):
            real_vals = real_mat[:, i]
            ai_vals = ai_mat[:, i]
            pooled_std = np.sqrt(
                (np.var(real_vals) + np.var(ai_vals)) / 2 + 1e-10
            )
            cohens_d = abs(np.mean(real_vals) - np.mean(ai_vals)) / pooled_std
            if len(real_vals) >= 2 and len(ai_vals) >= 2:
                _, p_val = stats.ttest_ind(real_vals, ai_vals, equal_var=False)
            else:
                p_val = 1.0
            results.append((i, cohens_d, p_val))

        results.sort(key=lambda x: -x[1])
        importance[tier_name] = results

    return importance


def print_top_features(
    importance: dict[str, list[tuple[int, float, float]]],
    top_k: int = 10,
) -> None:
    print("\n" + "=" * 60)
    print("  TOP DISCRIMINATIVE FEATURES (by Cohen's d)")
    print("=" * 60)
    for tier_name, features in importance.items():
        print(f"\n--- {tier_name.upper()} (top {top_k}) ---")
        print(f"  {'Idx':>4}  {'Cohen d':>9}  {'p-value':>10}")
        print(f"  {'---':>4}  {'---------':>9}  {'----------':>10}")
        for idx, d, p in features[:top_k]:
            sig = "*" if p < 0.05 else " "
            print(f"  {idx:>4}  {d:>9.4f}  {p:>10.4e} {sig}")


def plot_feature_distributions(
    feat_matrices: dict[str, dict[str, np.ndarray]],
    importance: dict[str, list[tuple[int, float, float]]],
) -> None:
    for tier_name in ["tier1", "tier2"]:
        top_feats = importance[tier_name][:6]
        n_plots = len(top_feats)
        if n_plots == 0:
            continue

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.ravel()

        real_mat = feat_matrices["real"][tier_name]
        ai_mat = feat_matrices["ai"][tier_name]

        for ax_idx, (feat_idx, d_val, p_val) in enumerate(top_feats):
            if ax_idx >= len(axes):
                break
            ax = axes[ax_idx]
            ax.hist(real_mat[:, feat_idx], bins=15, alpha=0.6, label="Real",
                    color="#2196F3", edgecolor="black")
            ax.hist(ai_mat[:, feat_idx], bins=15, alpha=0.6, label="AI",
                    color="#F44336", edgecolor="black")
            ax.set_title(f"Feature {feat_idx}\nd={d_val:.2f}, p={p_val:.2e}")
            ax.legend(fontsize=8)

        for ax_idx in range(n_plots, len(axes)):
            axes[ax_idx].set_visible(False)

        plt.suptitle(f"{tier_name.upper()} — Top Discriminative Features", fontsize=14)
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / f"features_{tier_name}_distributions.png", dpi=150)
        plt.close(fig)
        logger.info("Saved %s feature distribution plot", tier_name)


def plot_feature_heatmap(
    feat_matrices: dict[str, dict[str, np.ndarray]],
) -> None:
    for tier_name in ["tier1", "tier2"]:
        real_mean = feat_matrices["real"][tier_name].mean(axis=0)
        ai_mean = feat_matrices["ai"][tier_name].mean(axis=0)
        diff = np.abs(real_mean - ai_mean)

        fig, ax = plt.subplots(figsize=(max(12, len(diff) // 5), 3))
        n = len(diff)
        ax.bar(range(n), diff, color=sns.color_palette("coolwarm", n))
        ax.set_title(f"{tier_name.upper()} — Absolute Mean Difference (Real vs AI)")
        ax.set_xlabel("Feature Index")
        ax.set_ylabel("|mean_real - mean_ai|")
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / f"features_{tier_name}_mean_diff.png", dpi=150)
        plt.close(fig)
        logger.info("Saved %s mean-diff bar plot", tier_name)


# ── main ──────────────────────────────────────────────────────

def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  02 — FEATURE ENGINEERING ANALYSIS")
    print("=" * 60)

    print("\n[1/5] Loading audio samples ...")
    pool = try_load_audio_pool()
    if pool:
        print(f"  Loaded real dataset audio: {len(pool['real'])} real, "
              f"{len(pool['ai'])} AI tracks")
    else:
        print("  No dataset audio available. Generating synthetic signals ...")
        pool = generate_sample_pool(n_per_class=5)
        print(f"  Generated {len(pool['real'])} real-like and "
              f"{len(pool['ai'])} AI-like synthetic signals")

    print("\n[2/5] Extracting features ...")
    feat_matrices = extract_feature_matrix(pool)
    print_feature_stats(feat_matrices)

    print("\n[3/5] Computing feature importance ...")
    importance = compute_feature_importance(feat_matrices)
    print_top_features(importance)

    print("\n[4/5] Generating distribution plots ...")
    plot_feature_distributions(feat_matrices, importance)

    print("\n[5/5] Generating heatmap / bar plots ...")
    plot_feature_heatmap(feat_matrices)

    print(f"\n  All figures saved to: {FIGURES_DIR}")
    print("\n" + "=" * 60)
    print("  FEATURE ANALYSIS COMPLETE")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
