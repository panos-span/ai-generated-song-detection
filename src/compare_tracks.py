"""Main entry point for pairwise audio similarity comparison."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import librosa
import numpy as np
import torch
from scipy.ndimage import zoom
from scipy.spatial.distance import cosine as cosine_distance

from src.features.audio_features import (
    extract_all_features,
    extract_tier2_features,
    load_audio,
)
from src.features.embedding_features import EmbeddingExtractor
from src.models.chunking import chunk_audio
from src.models.similarity_head import PairwiseSimilarityModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Similarity dimension functions
# ---------------------------------------------------------------------------


def melodic_similarity(audio_a: np.ndarray, audio_b: np.ndarray, sr: int) -> float:
    """Chroma-based DTW distance between two tracks. Returns similarity in [0,1]."""
    chroma_a = librosa.feature.chroma_stft(y=audio_a, sr=sr, n_chroma=12)
    chroma_b = librosa.feature.chroma_stft(y=audio_b, sr=sr, n_chroma=12)

    # Sanitize NaN values (occur with silent/near-silent audio where
    # librosa cannot estimate pitch tuning from an empty frequency set)
    np.nan_to_num(chroma_a, copy=False, nan=0.0)
    np.nan_to_num(chroma_b, copy=False, nan=0.0)

    # Replace zero-norm columns with a tiny epsilon so cosine distance
    # does not produce NaN (0/0 division).
    eps = 1e-10
    for chroma in (chroma_a, chroma_b):
        norms = np.linalg.norm(chroma, axis=0)
        zero_cols = norms < eps
        if zero_cols.any():
            chroma[:, zero_cols] = eps

    # If either track has fewer than 2 frames, DTW is meaningless
    if chroma_a.shape[1] < 2 or chroma_b.shape[1] < 2:
        return 0.0

    # DTW returns (D, wp) where D is the accumulated cost matrix
    D, _wp = librosa.sequence.dtw(chroma_a, chroma_b, metric="cosine")
    # Normalised path cost: last cell / path length
    path_cost = D[-1, -1]
    path_length = D.shape[0] + D.shape[1]
    normalised = path_cost / max(path_length, 1)

    # Cosine metric gives values in [0, 2]; normalised cost is in a similar range.
    # Map to [0,1] where 1 = identical melody.
    similarity = float(np.clip(1.0 - normalised, 0.0, 1.0))
    return similarity


def timbral_similarity(audio_a: np.ndarray, audio_b: np.ndarray, sr: int) -> float:
    """Cosine similarity of averaged MFCC vectors. Returns similarity in [0,1]."""
    mfcc_a = librosa.feature.mfcc(y=audio_a, sr=sr, n_mfcc=20)
    mfcc_b = librosa.feature.mfcc(y=audio_b, sr=sr, n_mfcc=20)

    mean_a = np.nan_to_num(np.mean(mfcc_a, axis=1), nan=0.0)
    mean_b = np.nan_to_num(np.mean(mfcc_b, axis=1), nan=0.0)

    # Zero vectors → no timbral information → return 0 similarity
    if np.linalg.norm(mean_a) == 0.0 or np.linalg.norm(mean_b) == 0.0:
        return 0.0

    # cosine_distance returns 1 - cos_sim, so cos_sim = 1 - dist
    dist = cosine_distance(mean_a, mean_b)
    similarity = float(np.clip(1.0 - dist, 0.0, 1.0))
    return similarity


def structural_similarity(audio_a: np.ndarray, audio_b: np.ndarray, sr: int) -> float:
    """Cross-correlation of Self-Similarity Matrices. Returns similarity in [0,1]."""
    chroma_a = librosa.feature.chroma_stft(y=audio_a, sr=sr, n_chroma=12)
    chroma_b = librosa.feature.chroma_stft(y=audio_b, sr=sr, n_chroma=12)

    # Sanitize NaN values from silent/near-silent audio
    np.nan_to_num(chroma_a, copy=False, nan=0.0)
    np.nan_to_num(chroma_b, copy=False, nan=0.0)

    # Build normalised SSMs
    chroma_a_norm = librosa.util.normalize(chroma_a, axis=0)
    chroma_b_norm = librosa.util.normalize(chroma_b, axis=0)
    ssm_a = chroma_a_norm.T @ chroma_a_norm
    ssm_b = chroma_b_norm.T @ chroma_b_norm

    # Resize both SSMs to the same fixed size for comparable correlation
    target_size = min(ssm_a.shape[0], ssm_b.shape[0], 200)
    if target_size < 2:
        return 0.0

    ssm_a_resized = zoom(ssm_a, target_size / np.array(ssm_a.shape), order=1)
    ssm_b_resized = zoom(ssm_b, target_size / np.array(ssm_b.shape), order=1)

    # Normalised cross-correlation
    a_flat = ssm_a_resized.ravel()
    b_flat = ssm_b_resized.ravel()

    a_centered = a_flat - a_flat.mean()
    b_centered = b_flat - b_flat.mean()
    norm_a = np.linalg.norm(a_centered)
    norm_b = np.linalg.norm(b_centered)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    ncc = float(np.dot(a_centered, b_centered) / (norm_a * norm_b))
    # Map from [-1, 1] to [0, 1]
    similarity = float(np.clip((ncc + 1.0) / 2.0, 0.0, 1.0))
    return similarity


def embedding_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Cosine similarity of embedding vectors. Returns similarity in [0,1]."""
    dist = cosine_distance(emb_a, emb_b)
    return float(np.clip(1.0 - dist, 0.0, 1.0))


# ---------------------------------------------------------------------------
# AI-artifact scoring
# ---------------------------------------------------------------------------


def compute_ai_artifact_score(audio: np.ndarray, sr: int) -> float:
    """Score indicating how likely a track contains AI generation artifacts.

    Uses Tier-2 features (phase, HNR, spectral flatness, rolloff) and
    Fourier artifact detection (deconvolution-induced spectral peaks from
    Afchar et al. ISMIR 2025). Fourier features are weighted highest as
    they provide the strongest single-feature separability (>90%).

    Returns score in [0,1] where 1 = very likely AI-generated.
    """
    tier2 = extract_tier2_features(audio, sr)
    # tier2 layout: [phase(7), hnr, sf_mean, sf_std, rolloff, ssm, fourier(10)]
    pci = tier2[0]
    hnr = tier2[7]
    sf_mean = tier2[8]
    rolloff_ratio = tier2[10]
    # Fourier artifacts start at index 12
    fourier_feats = tier2[12:22]

    scores: list[float] = []

    pci_score = float(np.clip((pci - 1.0) / 1.5, 0.0, 1.0))
    scores.append(pci_score)

    hnr_score = float(np.clip(1.0 - hnr / 20.0, 0.0, 1.0))
    scores.append(hnr_score)

    sf_score = float(np.clip(sf_mean * 5.0, 0.0, 1.0))
    scores.append(sf_score)

    ideal_rolloff = 0.87
    rolloff_dev = abs(rolloff_ratio - ideal_rolloff)
    rolloff_score = float(np.clip(rolloff_dev / 0.3, 0.0, 1.0))
    scores.append(rolloff_score)

    # Fourier artifact composite: average of peak-to-background ratios,
    # periodicity, and artifact energy ratio
    fourier_score = float(np.mean(fourier_feats)) if len(fourier_feats) == 10 else 0.0
    scores.append(fourier_score)

    # Fourier features weighted highest (strongest indicator per A2 paper)
    weights = [0.15, 0.15, 0.10, 0.10, 0.50]
    artifact_score = float(np.dot(scores, weights))
    return float(np.clip(artifact_score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Neural model scoring (optional)
# ---------------------------------------------------------------------------


def _pad_or_truncate(features: np.ndarray, max_chunks: int = 12) -> np.ndarray:
    """Ensure exactly *max_chunks* rows via zero-padding or truncation."""
    n_chunks, feat_dim = features.shape
    if n_chunks >= max_chunks:
        return features[:max_chunks]
    padded = np.zeros((max_chunks, feat_dim), dtype=features.dtype)
    padded[:n_chunks] = features
    return padded


def _neural_similarity(
    audio_a: np.ndarray,
    audio_b: np.ndarray,
    sr: int,
    model_path: str,
    device: str,
    feature_stats_path: str | None = None,
) -> float | None:
    """Load a trained PairwiseSimilarityModel and compute a neural similarity score.

    Returns None if the model cannot be loaded or inference fails.
    """
    try:
        if device == "auto":
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            dev = torch.device(device)

        checkpoint = torch.load(model_path, map_location=dev, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        # Infer feature_dim from checkpoint
        proj_weight_key = "siamese.projector.net.0.weight"
        if proj_weight_key in state_dict:
            feature_dim = state_dict[proj_weight_key].shape[1]
        else:
            feature_dim = 452  # default combined tier1+tier2

        model = PairwiseSimilarityModel(feature_dim=feature_dim)
        model.load_state_dict(state_dict, strict=False)
        model.to(dev).eval()

        # Per-chunk feature extraction (matches training pipeline)
        chunks_a = chunk_audio(audio_a, sr)
        chunks_b = chunk_audio(audio_b, sr)

        feats_a = np.stack(
            [extract_all_features(c, sr)["combined"] for c in chunks_a], axis=0
        )  # (n_chunks_a, feature_dim)
        feats_b = np.stack(
            [extract_all_features(c, sr)["combined"] for c in chunks_b], axis=0
        )  # (n_chunks_b, feature_dim)

        # Z-score normalization (match training standardization)
        if feature_stats_path is not None:
            stats = np.load(feature_stats_path)
            feat_mean, feat_std = stats["mean"], stats["std"]
            feats_a = (feats_a - feat_mean) / (feat_std + 1e-8)
            feats_b = (feats_b - feat_mean) / (feat_std + 1e-8)

        # Pad/truncate to fixed chunk count (matches training)
        feats_a = _pad_or_truncate(feats_a)
        feats_b = _pad_or_truncate(feats_b)

        t_a = torch.from_numpy(feats_a).float().unsqueeze(0).to(dev)
        t_b = torch.from_numpy(feats_b).float().unsqueeze(0).to(dev)

        with torch.no_grad():
            score = model(t_a, t_b)

        return float(score.item())

    except Exception:
        logger.warning(
            "Neural model scoring failed; falling back to handcrafted features", exc_info=True
        )
        return None


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS = {
    "melodic": 0.15,
    "timbral": 0.15,
    "structural": 0.10,
    "embedding": 0.15,
    "neural": 0.30,
    "lyrical": 0.10,
    "vocal": 0.05,
}

ATTRIBUTION_THRESHOLD = 0.65


def _to_native(obj: object) -> object:
    """Recursively convert numpy types to JSON-serialisable Python types."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def compare_tracks(
    track_a_path: str,
    track_b_path: str,
    model_path: str | None = None,
    use_embeddings: bool = True,
    use_lyrics: bool = True,
    use_vocals: bool = True,
    device: str = "auto",
    mode: str = "standard",
    feature_stats_path: str | None = None,
) -> dict:
    """Compare two audio tracks and return attribution analysis.

    Args:
        mode: Inference mode -- "fast", "standard", or "full".
            fast: Fourier artifact scan only (early exit if strong AI signature).
            standard: Fourier + lyrics/speech + heuristic embedding similarity.
            full: All modalities including dual-stream neural + segment transformer.

    Returns a dict with per-dimension similarities, composite attribution
    score, AI-artifact scores, and a human-readable confidence label.
    """
    logger.info("Loading audio: %s", track_a_path)
    audio_a, sr = load_audio(track_a_path)
    logger.info("Loading audio: %s", track_b_path)
    audio_b, sr_b = load_audio(track_b_path, sr=sr)

    # -- Fast mode: Fourier artifact scan only --
    if mode == "fast":
        artifact_a = compute_ai_artifact_score(audio_a, sr)
        artifact_b = compute_ai_artifact_score(audio_b, sr)
        max_artifact = max(artifact_a, artifact_b)
        return _to_native(
            {
                "attribution_score": max_artifact,
                "ai_artifact_score": {"track_a": artifact_a, "track_b": artifact_b},
                "is_likely_attribution": max_artifact > ATTRIBUTION_THRESHOLD,
                "confidence": (
                    "high" if max_artifact > 0.8 else "medium" if max_artifact > 0.5 else "low"
                ),
                "mode": "fast",
            }
        )

    # -- Handcrafted similarity dimensions --
    logger.info("Computing melodic similarity ...")
    mel_sim = melodic_similarity(audio_a, audio_b, sr)

    logger.info("Computing timbral similarity ...")
    tim_sim = timbral_similarity(audio_a, audio_b, sr)

    logger.info("Computing structural similarity ...")
    str_sim = structural_similarity(audio_a, audio_b, sr)

    # -- Embedding similarity (optional) --
    emb_scores: dict[str, float] = {}
    if use_embeddings:
        try:
            logger.info("Extracting MERT / CLAP embeddings ...")
            extractor = EmbeddingExtractor(device=device)

            embs_a = extractor.extract_all_embeddings(audio_a, sr)
            embs_b = extractor.extract_all_embeddings(audio_b, sr)

            for key in embs_a:
                emb_scores[key] = embedding_similarity(embs_a[key], embs_b[key])
        except Exception:
            logger.warning(
                "Embedding extraction failed; continuing without embeddings", exc_info=True
            )
            use_embeddings = False

    # -- Lyrical similarity (Phase 2) --
    lyrical_score: float | None = None
    if use_lyrics:
        try:
            from src.features.lyrics_features import compute_lyrical_similarity

            logger.info("Computing lyrical similarity ...")
            lyrical_score = compute_lyrical_similarity(track_a_path, track_b_path, device=device)
        except Exception:
            logger.warning("Lyrical analysis failed; continuing without lyrics", exc_info=True)

    # -- Vocal similarity (Phase 2) --
    vocal_score: float | None = None
    if use_vocals:
        try:
            from src.features.speech_features import compute_vocal_similarity

            logger.info("Computing vocal similarity ...")
            vocal_score = compute_vocal_similarity(track_a_path, track_b_path, device=device)
        except Exception:
            logger.warning("Vocal analysis failed; continuing without vocals", exc_info=True)

    # -- Neural model score (optional, full mode or if model_path given) --
    neural_score: float | None = None
    if model_path is not None:
        logger.info("Running neural model scoring ...")
        neural_score = _neural_similarity(
            audio_a, audio_b, sr, model_path, device,
            feature_stats_path=feature_stats_path,
        )

    # -- AI artifact scores --
    logger.info("Computing AI-artifact scores ...")
    artifact_a = compute_ai_artifact_score(audio_a, sr)
    artifact_b = compute_ai_artifact_score(audio_b, sr)

    # -- Composite attribution score --
    weights = dict(_DEFAULT_WEIGHTS)

    # Average embedding scores into a single embedding component
    if emb_scores:
        avg_emb: float | None = float(np.mean(list(emb_scores.values())))
    else:
        avg_emb = None

    if neural_score is None:
        weights.pop("neural", None)
    if avg_emb is None:
        weights.pop("embedding", None)
    if lyrical_score is None:
        weights.pop("lyrical", None)
    if vocal_score is None:
        weights.pop("vocal", None)

    # Re-normalise weights to sum to 1
    total_w = sum(weights.values())
    weights = {k: v / total_w for k, v in weights.items()}

    component_scores = {
        "melodic": mel_sim,
        "timbral": tim_sim,
        "structural": str_sim,
    }
    if avg_emb is not None:
        component_scores["embedding"] = avg_emb
    if neural_score is not None:
        component_scores["neural"] = neural_score
    if lyrical_score is not None:
        component_scores["lyrical"] = lyrical_score
    if vocal_score is not None:
        component_scores["vocal"] = vocal_score

    attribution_score = sum(weights[k] * component_scores[k] for k in weights)

    # -- Confidence level --
    values = list(component_scores.values())
    component_std = float(np.std(values)) if len(values) > 1 else 0.0
    score_magnitude = abs(attribution_score - 0.5)

    if score_magnitude > 0.25 and component_std < 0.15:
        confidence = "high"
    elif score_magnitude > 0.12 or component_std < 0.20:
        confidence = "medium"
    else:
        confidence = "low"

    result: dict = {
        "attribution_score": attribution_score,
        "melodic_similarity": mel_sim,
        "timbral_similarity": tim_sim,
        "structural_similarity": str_sim,
        "embedding_similarity": emb_scores if emb_scores else None,
        "lyrical_similarity": lyrical_score,
        "vocal_similarity": vocal_score,
        "neural_similarity": neural_score,
        "ai_artifact_score": {
            "track_a": artifact_a,
            "track_b": artifact_b,
        },
        "is_likely_attribution": attribution_score > ATTRIBUTION_THRESHOLD,
        "confidence": confidence,
        "mode": mode,
        "details": {
            "weights_used": weights,
            "component_scores": component_scores,
            "component_agreement_std": component_std,
            "threshold": ATTRIBUTION_THRESHOLD,
        },
    }
    return _to_native(result)


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

_DIVIDER = "=" * 60


def _pretty_print(result: dict) -> None:
    """Print comparison results in a human-readable format."""
    print(f"\n{_DIVIDER}")
    print("  AUDIO SIMILARITY / ATTRIBUTION REPORT")
    print(_DIVIDER)

    score = result["attribution_score"]
    verdict = "YES" if result["is_likely_attribution"] else "NO"
    conf = result["confidence"].upper()

    print(f"\n  Attribution Score : {score:.4f}")
    print(f"  Likely Attribution: {verdict}  (confidence: {conf})")

    print(f"\n{chr(0x2500) * 60}")
    print("  Similarity Dimensions:")
    print(f"    Melodic    : {result['melodic_similarity']:.4f}")
    print(f"    Timbral    : {result['timbral_similarity']:.4f}")
    print(f"    Structural : {result['structural_similarity']:.4f}")

    if result.get("embedding_similarity"):
        for name, val in result["embedding_similarity"].items():
            print(f"    Embedding ({name:>5s}) : {val:.4f}")

    if result.get("lyrical_similarity") is not None:
        print(f"    Lyrical    : {result['lyrical_similarity']:.4f}")

    if result.get("vocal_similarity") is not None:
        print(f"    Vocal      : {result['vocal_similarity']:.4f}")

    if result.get("neural_similarity") is not None:
        print(f"    Neural     : {result['neural_similarity']:.4f}")

    print(f"\n{chr(0x2500) * 60}")
    print("  AI-Artifact Scores:")
    art = result["ai_artifact_score"]
    print(f"    Track A : {art['track_a']:.4f}")
    print(f"    Track B : {art['track_b']:.4f}")

    print(f"\n{chr(0x2500) * 60}")
    print("  Weights Used:")
    for k, v in result["details"]["weights_used"].items():
        print(f"    {k:>12s} : {v:.3f}")

    print(f"\n{_DIVIDER}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compare two audio tracks for AI attribution similarity",
    )
    parser.add_argument("track_a", help="Path to first audio track")
    parser.add_argument("track_b", help="Path to second audio track")
    parser.add_argument("--model", default=None, help="Path to trained model checkpoint")
    parser.add_argument(
        "--no-embeddings", action="store_true", help="Skip MERT/CLAP embeddings (faster)"
    )
    parser.add_argument("--no-lyrics", action="store_true", help="Skip lyrical analysis")
    parser.add_argument("--no-vocals", action="store_true", help="Skip vocal analysis")
    parser.add_argument(
        "--mode",
        default="standard",
        choices=["fast", "standard", "full"],
        help="Inference mode: fast (artifact only), standard, full (all modalities)",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--feature-stats",
        default=None,
        help="Path to feature_stats.npz for z-score normalization (neural model)",
    )
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    parser.add_argument(
        "--json", action="store_true", dest="json_stdout", help="Print results as JSON to stdout"
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    result = compare_tracks(
        track_a_path=args.track_a,
        track_b_path=args.track_b,
        model_path=args.model,
        use_embeddings=not args.no_embeddings,
        use_lyrics=not args.no_lyrics,
        use_vocals=not args.no_vocals,
        device=args.device,
        mode=args.mode,
        feature_stats_path=args.feature_stats,
    )

    if args.json_stdout:
        print(json.dumps(result, indent=2))
    else:
        _pretty_print(result)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
