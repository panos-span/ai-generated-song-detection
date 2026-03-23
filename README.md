# AI-Generated Song Detection

An end-to-end system for detecting AI-generated music by comparing audio track
pairs and predicting whether they share a common origin (original vs AI-generated).
It combines classical audio features, AI-artifact detectors, deep learned embeddings
(CLAP), lyrical analysis, and vocal fingerprinting through a Siamese neural network
with attention-based pooling.

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                       # install core dependencies
uv sync --extra dev           # install dev extras (pytest, jupyter)
uv sync --extra demucs        # install optional source separation (Demucs)
```

## Data Download

Each dataset has a dedicated download script. Run them from the project root:

```bash
uv run python data/download_sonics.py --num_samples 500    # SONICS (HuggingFace)
uv run python data/download_fakemusiccaps.py --num_samples 1500  # FakeMusicCaps
uv run python data/download_mippia.py          # MIPPIA SMP
```

Downloaded audio and metadata are saved under `data/sonics/`, `data/fakemusiccaps/`,
and `data/mippia/` respectively.

### Dataset Summary

| Dataset | Samples | Notes |
|---------|---------|-------|
| SONICS | 500 | Real + AI (Suno, Udio) |
| MIPPIA | 122 tracks (70 pairs) | 5 orphaned files on disk but not in metadata CSV |
| FakeMusicCaps | 1,500 | 5 generators (300 per model) |
| **Total** | **2,122** | |

## Usage

### CLI

```bash
uv run python src/compare_tracks.py track_a.wav track_b.wav
```

Optional flags:

| Flag | Description |
|---|---|
| `--mode fast\|standard\|full` | Inference mode (default: `standard`) |
| `--no-lyrics` | Skip lyrical analysis |
| `--no-vocals` | Skip vocal analysis |
| `--no-embeddings` | Skip MERT/CLAP embeddings |
| `--model PATH` | Path to a trained model checkpoint |
| `--device cpu\|cuda\|auto` | Compute device (default: `auto`) |
| `--feature-stats PATH` | Path to `feature_stats.npz` for neural model z-score normalization |
| `--output results.json` | Save results to JSON |
| `--json` | Print JSON to stdout |
| `-v` | Verbose logging |

### Python API

```python
from src.compare_tracks import compare_tracks

result = compare_tracks(
    track_a_path="track_a.wav",
    track_b_path="track_b.wav",
    use_embeddings=True,
    use_lyrics=True,
    use_vocals=True,
    device="auto",
    mode="standard",
)
```

The result dictionary includes:

| Key | Description |
|---|---|
| `attribution_score` | Composite score in [0, 1] |
| `melodic_similarity` | Melodic similarity score |
| `timbral_similarity` | Timbral similarity score |
| `structural_similarity` | Structural similarity score |
| `embedding_similarity` | MERT/CLAP embedding similarity |
| `lyrical_similarity` | Lyrical similarity score |
| `vocal_similarity` | Vocal similarity score |
| `neural_similarity` | Neural network similarity |
| `ai_artifact_score` | AI artifact detection score |
| `is_likely_attribution` | `True` if score exceeds threshold |
| `confidence` | `"high"`, `"medium"`, or `"low"` |
| `mode` | Inference mode used |

## Training

### 1. Construct training pairs

```bash
uv run python -m src.models.construct_pairs --data_dir data --output_dir data/pairs
```

### 2. Pre-compute features

```bash
uv run python data/precompute_features.py --skip_hnr --max_workers 6 --n_augmentations 5
```

### 3. Train the model

```bash
uv run python -m src.models.train --pairs_csv data/pairs/train_pairs.csv \
  --epochs 50 --batch_size 16 --augment --prefetch \
  --feature_cache_dir data/feature_cache
```

Additional training flags:

| Flag | Description |
|---|---|
| `--augment` | Enable data augmentation |
| `--aug_prob` | Augmentation probability |
| `--prefetch` | Pre-fetch features into memory |
| `--prefetch_workers` | Number of prefetch workers |
| `--dual-stream` | Use DualStream architecture (optional, not used by default) |
| `--use-segment-transformer` | Enable segment transformer (optional, not used by default) |
| `--triplet-weight` | Weight for triplet loss |
| `--load-pretrained` | Load pre-trained encoder weights |

### 4. Pre-training (SimCLR)

```bash
uv run python -m src.models.pretrain --data_dir data --epochs 100
```

### 5. Knowledge distillation

```bash
uv run python -m src.models.distill --teacher_path models/best_model.pt --data_dir data
```

### 6. Hyperparameter tuning

```bash
uv run python -m src.models.tune --pairs_csv data/pairs/train_pairs.csv
```

## Evaluation

Run the standalone evaluation on the held-out test set:

```bash
uv run python -m src.models.evaluate --model models/best_model.pt \
  --pairs_csv data/pairs/test_pairs.csv \
  --feature_cache_dir data/feature_cache
```

This computes classification metrics (accuracy, precision, recall, F1, AUC-ROC),
plots the ROC curve and confusion matrix, and prints per-pair scores.
Test-set result: **510 pairs, AUC = 0.9981, F1 = 0.98**.

## Testing

```bash
uv run pytest
```

34 unit tests cover feature extraction, chunking, model shapes, similarity
functions, feature statistics, random chunk sampling, and edge cases
(silence, short audio).

## Project Structure

```
src/
  __init__.py
  log_config.py                    # Loguru configuration
  compare_tracks.py                # CLI entry point and compare_tracks() API
  features/
    __init__.py
    audio_features.py              # Tier 1 + Tier 2 + Fourier artifact detection
    embedding_features.py          # MERT + CLAP embeddings
    augmentations.py               # 11 augmentation types
    gpu_features.py                # GPU-accelerated features (torchaudio + nnAudio)
    streaming.py                   # Streaming extraction for large files (stub)
    lyrics_features.py             # Whisper transcription + SBERT embeddings (inference-only)
    speech_features.py             # Wav2Vec2 + Whisper encoder + Silero VAD (inference-only)
    source_separation.py           # Optional Demucs vocal isolation (stub)
  models/
    __init__.py
    chunking.py                    # Sliding-window audio chunking
    siamese_network.py             # Siamese + DualStream + SegmentTransformer + GatedFusion
    spectrogram_encoder.py         # CNN spectrogram encoder
    similarity_head.py             # SimilarityHead + CoarseToFineHead
    pair_dataset.py                # Dataset + prefetch_features()
    train.py                       # Training pipeline + CLI
    construct_pairs.py             # Pair construction
    evaluate.py                    # Standalone evaluation with ROC/confusion matrix plots
    tune.py                        # Optuna hyperparameter tuning (not executed)
    pretrain.py                    # SimCLR contrastive pre-training (not executed)
    distill.py                     # Knowledge distillation (not executed)
data/
  download_sonics.py               # SONICS dataset downloader
  download_fakemusiccaps.py        # FakeMusicCaps downloader
  download_mippia.py               # MIPPIA SMP downloader
  precompute_features.py           # Feature pre-computation
  validate_datasets.py             # Dataset validation
notebooks/
  01_eda.py                        # EDA script
  02_feature_analysis.py           # Feature analysis script
  03_evaluation.py                 # Evaluation pipeline
  figures/                         # 40 EDA and evaluation figures
tests/
  test_core.py                     # 34 unit tests
report/
  report.md                        # Technical report (Markdown)
  report.tex                       # Technical report (LaTeX)
  report.pdf                       # Compiled report (36 pages)
PIPELINE.md                        # End-to-end pipeline runbook
```

## Key Dependencies

| Library | Purpose |
|---|---|
| librosa | Audio loading, MFCCs, chroma, spectral features |
| torch / torchaudio | Neural network training and audio resampling |
| transformers | MERT and CLAP pre-trained model inference |
| faster-whisper | Whisper ASR for lyrical analysis |
| sentence-transformers | SBERT text embeddings |
| nnAudio | GPU-accelerated STFT/Mel/MFCC |
| datasets | HuggingFace dataset streaming (SONICS) |
| scikit-learn | Train/val/test splitting, AUC-ROC metrics |
| scipy | DTW, cosine distance, image zoom for SSM |
| mlflow | Experiment tracking |
| loguru | Structured logging |
| optuna | Hyperparameter tuning |
| pandas | Metadata and pair CSV management |
| numpy | Numerical operations throughout |
| soundfile | WAV file I/O |
| faiss-cpu | Fast nearest-neighbour search |
| matplotlib / seaborn | Visualisation in notebooks |
