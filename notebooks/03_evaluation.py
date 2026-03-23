"""Evaluation Pipeline for AI Audio Similarity project.

Loads test pairs, runs the comparison pipeline, computes classification
metrics, and generates evaluation plots. Falls back to synthetic test
pairs if no real data is available.

Usage:
    uv run python notebooks/03_evaluation.py
"""
from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import soundfile as sf
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.compare_tracks import compare_tracks

FIGURES_DIR = PROJECT_ROOT / "notebooks" / "figures"
REPORT_DIR = PROJECT_ROOT / "report"
DATA_DIR = PROJECT_ROOT / "data"
SR = 16000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", palette="muted")


# ── synthetic pair generation ──────────────────────────────────

def _make_real_signal(duration: float = 3.0, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    freq = 220 + seed * 50
    sig = (
        0.5 * np.sin(2 * np.pi * freq * t)
        + 0.3 * np.sin(2 * np.pi * freq * 2 * t)
        + 0.1 * np.sin(2 * np.pi * freq * 3 * t)
        + 0.03 * rng.randn(len(t))
    )
    envelope = np.exp(-0.2 * t) * (1 + 0.2 * np.sin(2 * np.pi * 3 * t))
    return (sig * envelope).astype(np.float32)


def _make_ai_signal(duration: float = 3.0, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed + 1000)
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    freq = 220 + seed * 50
    sig = (
        0.4 * np.sin(2 * np.pi * freq * t)
        + 0.4 * np.sin(2 * np.pi * freq * 2 * t)
        + 0.2 * rng.randn(len(t))
    )
    return sig.astype(np.float32)


def _make_derivative(original: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed + 500)
    stretched = np.interp(
        np.linspace(0, len(original) - 1, int(len(original) * 1.05)),
        np.arange(len(original)),
        original,
    )[:len(original)]
    return (stretched + 0.02 * rng.randn(len(stretched))).astype(np.float32)


def create_synthetic_test_pairs(
    tmp_dir: Path, n_pairs: int = 10,
) -> pd.DataFrame:
    rows: list[dict] = []
    audio_dir = tmp_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_pairs):
        if i < n_pairs // 2:
            real = _make_real_signal(duration=3.0, seed=i)
            deriv = _make_derivative(real, seed=i)
            path_a = audio_dir / f"real_{i}.wav"
            path_b = audio_dir / f"deriv_{i}.wav"
            sf.write(str(path_a), real, SR)
            sf.write(str(path_b), deriv, SR)
            rows.append({
                "track_a_path": str(path_a),
                "track_b_path": str(path_b),
                "label": 1,
                "pair_type": "positive",
            })
        else:
            real = _make_real_signal(duration=3.0, seed=i)
            ai = _make_ai_signal(duration=3.0, seed=i + 100)
            path_a = audio_dir / f"real_{i}.wav"
            path_b = audio_dir / f"ai_{i}.wav"
            sf.write(str(path_a), real, SR)
            sf.write(str(path_b), ai, SR)
            rows.append({
                "track_a_path": str(path_a),
                "track_b_path": str(path_b),
                "label": 0,
                "pair_type": "negative",
            })

    return pd.DataFrame(rows)


def load_test_pairs() -> pd.DataFrame | None:
    test_csv = DATA_DIR / "pairs" / "test.csv"
    if not test_csv.exists():
        logger.warning("Test pairs CSV not found at %s", test_csv)
        return None
    df = pd.read_csv(test_csv)
    if "track_a_path" not in df.columns or "label" not in df.columns:
        logger.warning("Test CSV missing required columns")
        return None
    valid = df[
        df["track_a_path"].apply(lambda p: Path(p).exists())
        & df["track_b_path"].apply(lambda p: Path(p).exists())
    ]
    if valid.empty:
        logger.warning("No valid audio paths in test CSV")
        return None
    logger.info("Loaded %d test pairs from %s", len(valid), test_csv)
    return valid


# ── evaluation ─────────────────────────────────────────────────

def run_evaluation(pairs_df: pd.DataFrame) -> dict:
    y_true: list[int] = []
    y_scores: list[float] = []
    pair_results: list[dict] = []

    total = len(pairs_df)
    for idx, row in pairs_df.iterrows():
        print(f"  Evaluating pair {idx + 1}/{total} ...", end="\r")
        try:
            result = compare_tracks(
                track_a_path=row["track_a_path"],
                track_b_path=row["track_b_path"],
                use_embeddings=False,
            )
            score = result["attribution_score"]
        except Exception as exc:
            logger.warning("Pair %d failed: %s", idx, exc)
            continue

        y_true.append(int(row["label"]))
        y_scores.append(float(score))
        pair_results.append({
            "track_a": row["track_a_path"],
            "track_b": row["track_b_path"],
            "label": int(row["label"]),
            "score": float(score),
            "predicted": int(score > 0.65),
        })

    print()
    return {"y_true": y_true, "y_scores": y_scores, "pairs": pair_results}


def compute_metrics(y_true: list[int], y_scores: list[float], threshold: float = 0.65) -> dict:
    y_pred = [1 if s > threshold else 0 for s in y_scores]
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "auc_roc": float(roc_auc),
        "threshold": threshold,
        "confusion_matrix": cm.tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "n_samples": len(y_true),
    }


def print_metrics(metrics: dict) -> None:
    print("\n" + "=" * 60)
    print("  EVALUATION METRICS")
    print("=" * 60)
    print(f"  Samples evaluated : {metrics['n_samples']}")
    print(f"  Threshold         : {metrics['threshold']:.2f}")
    print(f"  Accuracy          : {metrics['accuracy']:.4f}")
    print(f"  Precision         : {metrics['precision']:.4f}")
    print(f"  Recall            : {metrics['recall']:.4f}")
    print(f"  F1 Score          : {metrics['f1_score']:.4f}")
    print(f"  AUC-ROC           : {metrics['auc_roc']:.4f}")

    cm = metrics["confusion_matrix"]
    print("\n  Confusion Matrix:")
    print(f"                 Predicted Neg  Predicted Pos")
    print(f"  Actual Neg     {cm[0][0]:>12}  {cm[0][1]:>12}")
    print(f"  Actual Pos     {cm[1][0]:>12}  {cm[1][1]:>12}")


# ── plots ──────────────────────────────────────────────────────

def plot_roc_curve(metrics: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(metrics["fpr"], metrics["tpr"], linewidth=2,
            label=f"AUC = {metrics['auc_roc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Attribution Detection")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "eval_roc_curve.png", dpi=150)
    plt.close(fig)
    logger.info("Saved ROC curve plot")


def plot_score_distributions(
    y_true: list[int], y_scores: list[float], threshold: float = 0.65,
) -> None:
    pos_scores = [s for s, y in zip(y_scores, y_true) if y == 1]
    neg_scores = [s for s, y in zip(y_scores, y_true) if y == 0]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 25)
    if pos_scores:
        ax.hist(pos_scores, bins=bins, alpha=0.6, label="Positive (similar)",
                color="#2196F3", edgecolor="black")
    if neg_scores:
        ax.hist(neg_scores, bins=bins, alpha=0.6, label="Negative (different)",
                color="#F44336", edgecolor="black")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5,
               label=f"Threshold = {threshold:.2f}")
    ax.set_xlabel("Attribution Score")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution — Positive vs Negative Pairs")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "eval_score_distributions.png", dpi=150)
    plt.close(fig)
    logger.info("Saved score distribution plot")


def plot_confusion_matrix(metrics: dict) -> None:
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "eval_confusion_matrix.png", dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix plot")


# ── main ──────────────────────────────────────────────────────

def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  03 — EVALUATION PIPELINE")
    print("=" * 60)

    print("\n[1/5] Loading test pairs ...")
    pairs_df = load_test_pairs()
    tmp_dir_obj = None

    if pairs_df is None:
        print("  No real test pairs available. Creating synthetic test pairs ...")
        tmp_dir_obj = tempfile.TemporaryDirectory()
        tmp_path = Path(tmp_dir_obj.name)
        pairs_df = create_synthetic_test_pairs(tmp_path, n_pairs=10)
        print(f"  Created {len(pairs_df)} synthetic test pairs")

    print(f"\n  Pairs: {len(pairs_df)} total")
    if "pair_type" in pairs_df.columns:
        for pt, count in pairs_df["pair_type"].value_counts().items():
            print(f"    {pt}: {count}")

    print("\n[2/5] Running evaluation pipeline ...")
    eval_results = run_evaluation(pairs_df)

    if len(eval_results["y_true"]) < 2:
        print("  ERROR: Not enough valid results for metric computation.")
        if tmp_dir_obj:
            tmp_dir_obj.cleanup()
        return

    print("\n[3/5] Computing metrics ...")
    metrics = compute_metrics(eval_results["y_true"], eval_results["y_scores"])
    print_metrics(metrics)

    print("\n[4/5] Generating plots ...")
    plot_roc_curve(metrics)
    plot_score_distributions(
        eval_results["y_true"], eval_results["y_scores"],
    )
    plot_confusion_matrix(metrics)

    print("\n[5/5] Saving results ...")
    output = {
        "metrics": metrics,
        "pair_results": eval_results["pairs"],
    }
    results_path = REPORT_DIR / "evaluation_results.json"
    results_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"  Results saved to: {results_path}")

    if tmp_dir_obj:
        tmp_dir_obj.cleanup()

    print(f"\n  Figures saved to: {FIGURES_DIR}")
    print("\n" + "=" * 60)
    print("  EVALUATION COMPLETE")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
