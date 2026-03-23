"""Standalone model evaluation script.

Loads a trained PairwiseSimilarityModel checkpoint, runs inference on a
test-pair CSV, computes classification metrics, and generates three plots:
  - eval_confusion_matrix.png
  - eval_roc_curve.png
  - eval_score_distributions.png

Usage
-----
python -m src.models.evaluate \
    --model_path models/best_model.pt \
    --pairs_csv data/pairs/test_pairs.csv \
    --feature_cache_dir data/feature_cache \
    --output_dir notebooks/figures \
    --threshold 0.5 \
    --batch_size 16
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.pair_dataset import AudioPairDataset, collate_pairs
from src.models.similarity_head import PairwiseSimilarityModel

logger = logging.getLogger(__name__)

# ── Plotting style ────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 12,
})


def load_model(model_path: str, device: torch.device) -> PairwiseSimilarityModel:
    """Load a PairwiseSimilarityModel from a checkpoint file."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # torch.compile() prefixes keys with "_orig_mod." — strip for portability
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}

    proj_key = "siamese.projector.net.0.weight"
    feature_dim = state_dict[proj_key].shape[1] if proj_key in state_dict else 452

    model = PairwiseSimilarityModel(feature_dim=feature_dim)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def run_inference(
    model: PairwiseSimilarityModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run model inference and return (scores, labels) as numpy arrays."""
    all_scores: list[float] = []
    all_labels: list[float] = []
    use_amp = device.type == "cuda"

    for batch in tqdm(loader, desc="Evaluating"):
        feat_a = batch["features_a"].to(device)
        feat_b = batch["features_b"].to(device)
        labels = batch["label"]

        with torch.amp.autocast("cuda", enabled=use_amp):
            emb_a, emb_b = model.siamese(feat_a, feat_b)
            logits = model.head(emb_a, emb_b)

        scores = torch.sigmoid(logits).cpu().tolist()
        all_scores.extend(scores)
        all_labels.extend(labels.tolist())

    return np.array(all_scores), np.array(all_labels)


def plot_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    threshold: float,
    output_path: Path,
) -> None:
    """Plot and save a confusion matrix heatmap."""
    cm = confusion_matrix(labels, predictions)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Unrelated", "Related"],
        yticklabels=["Unrelated", "Related"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix (threshold = {threshold})")
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved confusion matrix → %s", output_path)


def plot_roc_curve(
    labels: np.ndarray,
    scores: np.ndarray,
    output_path: Path,
) -> float:
    """Plot and save a ROC curve. Returns the AUC value."""
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random baseline")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — PairwiseSimilarityModel")
    ax.legend(loc="lower right")
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved ROC curve → %s", output_path)
    return roc_auc


def plot_score_distributions(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    output_path: Path,
) -> None:
    """Plot and save overlapping score distribution histograms."""
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pos_scores, bins=30, alpha=0.6, color="steelblue", label=f"Positive (n={len(pos_scores)})")
    ax.hist(neg_scores, bins=30, alpha=0.6, color="salmon", label=f"Negative (n={len(neg_scores)})")
    ax.axvline(threshold, color="black", linestyle="--", lw=1.5, label=f"Threshold = {threshold}")
    ax.set_xlabel("Similarity Score")
    ax.set_ylabel("Count")
    ax.set_title("Score Distributions — Positive vs. Negative Pairs")
    ax.legend()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved score distributions → %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained PairwiseSimilarityModel")
    parser.add_argument("--model_path", type=str, default="models/best_model.pt",
                        help="Path to the model checkpoint")
    parser.add_argument("--pairs_csv", type=str, default="data/pairs/test_pairs.csv",
                        help="Path to the test pairs CSV")
    parser.add_argument("--feature_cache_dir", type=str, default="data/feature_cache",
                        help="Directory with cached .npy features")
    parser.add_argument("--output_dir", type=str, default="notebooks/figures",
                        help="Directory to save output figures")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold for confusion matrix")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cpu, or cuda")
    parser.add_argument("--save_json", type=str, default=None,
                        help="Optional path to save metrics as JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    # ── Device ────────────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Using device: %s", device)

    # ── Load model ────────────────────────────────────────────────────
    model = load_model(args.model_path, device)
    logger.info("Loaded model from %s", args.model_path)

    # ── Build DataLoader ──────────────────────────────────────────────
    ds_kwargs: dict = {}
    stats_file = Path(args.feature_cache_dir) / "feature_stats.npz"
    if stats_file.exists():
        ds_kwargs["feature_stats_path"] = str(stats_file)

    dataset = AudioPairDataset(
        args.pairs_csv,
        feature_cache_dir=args.feature_cache_dir,
        training=False,
        **ds_kwargs,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_pairs,
    )
    logger.info("Loaded %d pairs from %s", len(dataset), args.pairs_csv)

    # ── Inference ─────────────────────────────────────────────────────
    scores, labels = run_inference(model, loader, device)

    # ── Metrics ───────────────────────────────────────────────────────
    predictions = (scores >= args.threshold).astype(int)
    acc = accuracy_score(labels, predictions)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0,
    )
    roc_auc_val = roc_auc_score(labels, scores) if len(set(labels)) > 1 else 0.0

    print("\n" + "=" * 50)
    print("  Evaluation Results")
    print("=" * 50)
    print(f"  Pairs evaluated : {len(labels)}")
    print(f"  Threshold       : {args.threshold}")
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  Precision       : {prec:.4f}")
    print(f"  Recall          : {rec:.4f}")
    print(f"  F1 Score        : {f1:.4f}")
    print(f"  AUC-ROC         : {roc_auc_val:.4f}")
    print("=" * 50 + "\n")

    # ── Plots ─────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(labels, predictions, args.threshold, out_dir / "eval_confusion_matrix.png")
    roc_auc_plot = plot_roc_curve(labels, scores, out_dir / "eval_roc_curve.png")
    plot_score_distributions(labels, scores, args.threshold, out_dir / "eval_score_distributions.png")

    # ── Optional JSON dump ────────────────────────────────────────────
    if args.save_json:
        metrics = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "auc_roc": float(roc_auc_val),
            "threshold": args.threshold,
            "n_samples": int(len(labels)),
            "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
        }
        json_path = Path(args.save_json)
        json_path.write_text(json.dumps(metrics, indent=2))
        logger.info("Saved metrics JSON → %s", json_path)


if __name__ == "__main__":
    main()
