# Full Pipeline — Commands & Details

End-to-end guide for running the AI-Original Pairwise Audio Similarity system,
from environment setup through evaluation and inference.

---

## Prerequisites

- **Python 3.13+** and [uv](https://docs.astral.sh/uv/) package manager
- **Kaggle API credentials** at `~/.kaggle/kaggle.json` (for SONICS dataset)
- ~10 GB disk space for datasets + feature cache

---

## Phase 0 — Environment Setup

```bash
uv sync                       # core dependencies
uv sync --extra dev           # dev extras (pytest, jupyter, notebooks)
uv sync --extra demucs        # optional: source separation via Demucs
```

---

## Phase 1 — Download Datasets

All download scripts are idempotent — safe to re-run. Each script skips files
that already exist on disk.

```bash
# FakeMusicCaps — AI-generated music from 5 generative models (Zenodo)
# Downloads audio from MusicGen_medium, audioldm2, musicldm, mustango, stable_audio_open
uv run python data/download_fakemusiccaps.py --num_samples 1000 --output_dir data/fakemusiccaps

# MIPPIA / SMP — Music plagiarism pairs with relation labels (YouTube)
# Downloads paired tracks (original + comparison) with plag/plag_doubt/remake labels
uv run python data/download_mippia.py --output_dir data/mippia

# SONICS — Real vs AI-generated songs (Kaggle)
# Downloads fake MP3s individually via Kaggle API, converts to WAV
uv run python data/download_sonics.py --num_samples 500 --output_dir data/sonics
```

### What gets created

| Directory | Contents |
|---|---|
| `data/fakemusiccaps/audio/` | 1000 WAV files (200 per model) |
| `data/fakemusiccaps/metadata.csv` | Filename, model, caption columns |
| `data/mippia/audio/` | Paired audio tracks (WAV) |
| `data/mippia/metadata.csv` | pair_id, track_a, track_b, similarity_label |
| `data/mippia/smp_dataset/` | Original SMP CSV with segment timestamps |
| `data/sonics/audio/` | Fake + real WAV files |
| `data/sonics/metadata.csv` | Filename, label (real/fake), model |

---

## Phase 1b — Validate Downloaded Datasets

Check all downloaded audio files for integrity before constructing pairs.
The script uses `soundfile.info()` (header-only, no full decode) so it is fast
even across thousands of files.

```bash
# Check all datasets — report only, no changes made
uv run python data/validate_datasets.py --data_dir data

# Check a single dataset
uv run python data/validate_datasets.py --data_dir data --dataset fakemusiccaps

# Delete invalid files + rewrite metadata CSVs, then re-download
uv run python data/validate_datasets.py --data_dir data --fix
# After fixing, re-run the Phase 1 download commands above —
# they are idempotent and will skip files that already exist.
```

### What gets validated

| Check | Expected value |
|---|---|
| File exists on disk | Present |
| File size | > 0 bytes |
| WAV header parseable | Valid (soundfile can read info) |
| Sample rate | 16 000 Hz (`--expected_sr`) |
| Channels | 1 (mono) |
| Duration | > 0.1 s (`--min_duration`) |

### Exit codes & output

| Exit code | Meaning |
|---|---|
| `0` | All files valid |
| `1` | One or more invalid files found |

Each invalid file is printed with its category (`MISSING`, `CORRUPT`,
`WRONG FORMAT`). Files found on disk but **not** referenced in the metadata
CSV are reported as **ORPHANED** (e.g. staging or in-progress files) and are
**never** automatically deleted.

---

## Phase 2 — Construct Training Pairs

Builds train/val/test pair CSVs from all available datasets.

```bash
uv run python -m src.models.construct_pairs \
    --data_dir data \
    --output_dir data/pairs
```

### Pair types generated

| Pair Type | Label | Source |
|---|---|---|
| `fmc_negative` | 0 | FakeMusicCaps: same YouTube ID, real vs AI-generated |
| `fmc_positive` | 1 | FakeMusicCaps: same YouTube ID across different AI models |
| `sonics_negative` | 0 | SONICS: real vs fake pairs |
| `mippia_positive` | 1 | MIPPIA: plagiarism/remake pairs |

**Output:** `data/pairs/train_pairs.csv`, `data/pairs/val_pairs.csv`, `data/pairs/test_pairs.csv`
with columns: `path_a, path_b, label, pair_type, split`

---

## Phase 3 — Precompute Features

Extracts Tier 1 (classical) audio features for all tracks referenced in the
pair CSVs and caches them as `.npy` files. This significantly speeds up training.

```bash
uv run python data/precompute_features.py \
    --pairs_csv data/pairs/train_pairs.csv \
                data/pairs/val_pairs.csv \
                data/pairs/test_pairs.csv \
    --cache_dir data/feature_cache \
    --skip_hnr \
    --max_workers 6
```

### Pre-generate augmented variants (recommended)

Generates N augmented copies per **training** audio file so training never calls
librosa. Val/test files are cached raw-only (augmented variants are not used
during evaluation). Each variant applies random audio augmentations (pitch
shift, time stretch, noise, codec re-encoding, etc.) and caches the extracted
features.

```bash
uv run python data/precompute_features.py \
    --pairs_csv data/pairs/train_pairs.csv \
                data/pairs/val_pairs.csv \
                data/pairs/test_pairs.csv \
    --cache_dir data/feature_cache \
    --n_augmentations 5 \
    --skip_hnr \
    --max_workers 6
```

> With `--skip_hnr` and `--max_workers 6`, expect **~10 minutes** for ~1 400
> training files with 5 augmented variants on a 6–8 core CPU.

**Output:** `data/feature_cache/*.npy` — one file per unique audio track per variant
(variant 0 = raw, variants 1–N = augmented).

---

## Phase 4 — Train Siamese Network

Trains the Siamese network with attention-based pooling on the constructed pairs.

```bash
uv run python -m src.models.train \
    --pairs_csv data/pairs/train_pairs.csv \
    --val_csv data/pairs/val_pairs.csv \
    --epochs 50 \
    --batch_size 16 \
    --lr 1e-4 \
    --accumulation_steps 4 \
    --output_dir models \
    --feature_cache_dir data/feature_cache
```

### Key training flags

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 50 | Number of training epochs |
| `--batch_size` | 16 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--accumulation_steps` | 4 | Gradient accumulation steps (effective batch = batch_size × steps) |
| `--feature_cache_dir` | — | Path to precomputed feature cache (skips on-the-fly extraction) |
| `--output_dir` | models | Where to save `best_model.pt` |
| `--contrastive_weight` | 0.5 | Weight of cosine contrastive loss term |
| `--contrastive_margin` | 0.4 | Cosine contrastive loss margin (cosine distance range) |
| `--n_augmentations` | 0 | Number of pre-computed augmented variants (use with Phase 3 augmentation) |
| `--feature_noise_std` | 0.01 | Gaussian noise added to cached features (0=disabled) |
| `--feature_dropout_p` | 0.05 | Feature dropout probability (0=disabled) |
| `--focal-gamma` | 2.0 | Focal loss focusing parameter (higher = more focus on hard examples) |
| `--focal-alpha` | 0.25 | Focal loss class balance weight |
| `--label-smoothing` | 0.0 | Label smoothing factor (0=disabled, try 0.05) |
| `--no-focal` | — | Use BCEWithLogitsLoss instead of FocalLoss |
| `--warmup-epochs` | 2 | Linear warmup epochs before cosine annealing |
| `--ema-decay` | 0.0 | EMA decay for stable evaluation (0=disabled, try 0.999) |
| `--triplet-weight` | 0.0 | Triplet loss weight with online hard-negative mining (0=disabled) |
| `--triplet-margin` | 0.3 | Triplet loss margin |
| `--dropout` | 0.1 | Dropout rate (increase to 0.2 to reduce overfitting) |
| `--weight_decay` | 1e-2 | AdamW weight decay |
| `--load-pretrained` | — | Path to pre-trained checkpoint to initialize from |

#### Hybrid augmentation (recommended when features are pre-computed)

```bash
uv run python -m src.models.train \
    --pairs_csv data/pairs/train_pairs.csv \
    --val_csv data/pairs/val_pairs.csv \
    --epochs 50 \
    --batch_size 32 \
    --accumulation_steps 2 \
    --output_dir models \
    --feature_cache_dir data/feature_cache \
    --n_augmentations 5 \
    --no-augment
```

This samples from 6 cached variants (1 raw + 5 augmented) per audio file and
adds lightweight feature-space noise. Training never calls librosa.

Feature standardization is applied automatically: a `feature_stats.npz` file is
computed from the feature cache on first run and used to z-score normalise all
452 feature dimensions (MFCC, Mel, Chroma, HNR, etc.) to zero mean / unit variance.

#### Recommended: with anti-overfitting regularization

```bash
uv run python -m src.models.train \
    --pairs_csv data/pairs/train_pairs.csv \
    --val_csv data/pairs/val_pairs.csv \
    --epochs 50 \
    --batch_size 32 \
    --accumulation_steps 2 \
    --output_dir models \
    --feature_cache_dir data/feature_cache \
    --n_augmentations 5 \
    --no-augment \
    --ema-decay 0.999 \
    --label-smoothing 0.05 \
    --dropout 0.2 \
    --triplet-weight 0.2
```

**Output:** `models/best_model.pt` — best checkpoint by validation AUC.

---

## Phase 5 — Evaluate

Run evaluation on the held-out test set.

### 5a — Metrics only (via train.py)

```bash
uv run python -m src.models.train \
    --eval_only \
    --pairs_csv data/pairs/test_pairs.csv \
    --model_path models/best_model.pt \
    --feature_cache_dir data/feature_cache \
    --batch_size 16
```

**Output:** Prints test loss, accuracy, AUC-ROC to stdout.

### 5b — Metrics + Plots (via evaluate.py)

Generates confusion matrix, ROC curve, and score distribution figures.

```bash
uv run python -m src.models.evaluate \
    --model_path models/best_model.pt \
    --pairs_csv data/pairs/test_pairs.csv \
    --feature_cache_dir data/feature_cache \
    --output_dir notebooks/figures \
    --threshold 0.5 \
    --batch_size 16
```

| Flag | Default | Description |
|---|---|---|
| `--model_path` | — | Path to trained checkpoint (required) |
| `--pairs_csv` | — | Path to test pairs CSV (required) |
| `--feature_cache_dir` | — | Feature cache directory (for z-score stats) |
| `--output_dir` | `notebooks/figures` | Where to save PNG plots |
| `--threshold` | `0.5` | Decision threshold for confusion matrix |
| `--batch_size` | `16` | Inference batch size |
| `--device` | `auto` | `cuda` or `cpu` (auto-detects GPU) |

**Output:**

| File | Description |
|---|---|
| `notebooks/figures/eval_confusion_matrix.png` | Confusion matrix heatmap at the chosen threshold |
| `notebooks/figures/eval_roc_curve.png` | ROC curve with AUC annotation |
| `notebooks/figures/eval_score_distributions.png` | Score histograms for positive/negative pairs |
| `report/evaluation_results.json` | Full metrics + per-pair scores (JSON) |

Metrics (accuracy, AUC-ROC, precision, recall, F1) are also printed to stdout.

---

## Phase 6 (Optional) — Self-Supervised Pretraining

Pretrain the spectrogram encoder using contrastive learning on unlabelled audio,
before fine-tuning with Phase 4.

```bash
uv run python -m src.models.pretrain \
    --data_dir data \
    --output_dir models \
    --epochs 20 \
    --batch_size 32 \
    --lr 1e-4 \
    --aug_prob 0.5
```

**Output:** `models/pretrained_best.pt`

To use the pretrained encoder as initialization for Phase 4 training:

```bash
uv run python -m src.models.train \
    --pairs_csv data/pairs/train_pairs.csv \
    --val_csv data/pairs/val_pairs.csv \
    --feature_cache_dir data/feature_cache \
    --output_dir models \
    --load-pretrained models/pretrained_best.pt \
    --epochs 50
```

---

## Phase 7 (Optional) — Hyperparameter Tuning

Optuna-based search over learning rate, weight decay, embed/hidden dims,
dropout, contrastive margin/weight, batch size, and gradient clip norm.

```bash
uv run python -m src.models.tune \
    --pairs_csv data/pairs/train_pairs.csv \
    --val_csv data/pairs/val_pairs.csv \
    --n_trials 50 \
    --tuning_epochs 20 \
    --patience 5 \
    --feature_cache_dir data/feature_cache \
    --output_dir models/tune_checkpoints \
    --study_db models/tune_study.db
```

**Output:** `models/tune_checkpoints/trial_N/best_model.pt` for each trial,
Optuna study persisted at `models/tune_study.db` (resume interrupted runs by
re-running the same command). Best hyperparameters are printed to stdout.

To resume an interrupted study:

```bash
uv run python -m src.models.tune \
    --pairs_csv data/pairs/train_pairs.csv \
    --val_csv data/pairs/val_pairs.csv \
    --n_trials 50 \
    --feature_cache_dir data/feature_cache \
    --output_dir models/tune_checkpoints \
    --study_db models/tune_study.db
    # Optuna will continue from where it left off
```

---

## Phase 8 (Optional) — Knowledge Distillation

Distill a trained teacher model into a smaller, faster student model.

```bash
uv run python -m src.models.distill \
    --teacher_path models/best_model.pt \
    --data_dir data \
    --output_dir models \
    --epochs 30 \
    --batch_size 32 \
    --student_dim 128
```

**Output:** `models/student_best.pt` — a compact model with `student_dim=128`
embedding dimension (vs teacher's 256). Use like any other model checkpoint.

---

## Phase 9 — Inference (Compare Two Tracks)

Compare any two audio files and get a similarity/attribution score.

```bash
# Human-readable report (real SONICS track vs AI-generated SONICS track)
uv run python -m src.compare_tracks \
    "data/sonics/audio/real_-8xdDaRFdwc.wav" \
    "data/sonics/audio/fake_53654_suno_1.wav"

# JSON output — FakeMusicCaps real vs AI-generated
uv run python -m src.compare_tracks \
    "data/fakemusiccaps/audio/MusicGen_medium_01PzcPKT3_E.wav" \
    "data/fakemusiccaps/audio/audioldm2_-0Gj8-vB1q4.wav" \
    --json

# Standard mode with trained model — two different AI models of the same caption
uv run python -m src.compare_tracks \
    "data/fakemusiccaps/audio/musicldm_09lQmg2wvsY.wav" \
    "data/fakemusiccaps/audio/mustango_09lQmg2wvsY.wav" \
    --model models/best_model.pt \
    --json

# Full mode — all tiers + embeddings + lyrics + vocals
uv run python -m src.compare_tracks \
    "data/sonics/audio/real_393C3pr2ioY.wav" \
    "data/sonics/audio/fake_53654_suno_1.wav" \
    --mode full \
    --model models/best_model.pt \
    --json

# Fast mode — Fourier artifact scan only (AI-detection, no pair comparison)
uv run python -m src.compare_tracks \
    "data/sonics/audio/real_Ah0Ys50CqO8.wav" \
    "data/sonics/audio/fake_53654_suno_1.wav" \
    --mode fast

# MIPPIA plagiarism pair — compare original vs suspected plagiarism
uv run python -m src.compare_tracks \
    "data/mippia/audio/pair003_a_The Gap Band - Oops Upside Your Head.wav" \
    "data/mippia/audio/pair003_b_*.wav" \
    --model models/best_model.pt \
    --no-embeddings --no-lyrics

# Save JSON result to file
uv run python -m src.compare_tracks \
    "data/sonics/audio/real_-8xdDaRFdwc.wav" \
    "data/sonics/audio/fake_53654_suno_1.wav" \
    --model models/best_model.pt \
    --output results.json

# Verbose logging (shows per-feature scores)
uv run python -m src.compare_tracks \
    "data/fakemusiccaps/audio/stable_audio_open_06Brdf83RZE.wav" \
    "data/fakemusiccaps/audio/musicldm_06Brdf83RZE.wav" \
    --model models/best_model.pt \
    --verbose --json
```

### Inference modes

| Mode | Features Used | Speed |
|---|---|---|
| `fast` | Tier 1 classical features only | ~2s |
| `standard` | Tier 1 + Tier 2 AI-artifact features | ~10s |
| `full` | All tiers + embeddings + lyrics + vocals | ~60s+ |

---

## Quick Start (Minimal Run)

Copy-paste this block to run the entire pipeline end-to-end:

> **Note:** On a CPU-only machine, add `--device cpu` to all `train`, `tune`, `pretrain`,
> and `distill` commands below. See **Full Pipeline Refresh Run** for a complete
> CPU-optimised command set.

```bash
uv sync
uv run python data/download_fakemusiccaps.py --num_samples 1000 --output_dir data/fakemusiccaps
uv run python data/download_mippia.py --output_dir data/mippia
uv run python data/download_sonics.py --num_samples 500 --output_dir data/sonics
# Optional: validate all downloads; add --fix to delete corrupt files then re-run the downloads above
# uv run python data/validate_datasets.py --data_dir data --fix
uv run python -m src.models.construct_pairs --data_dir data --output_dir data/pairs
uv run python data/precompute_features.py --pairs_csv data/pairs/train_pairs.csv data/pairs/val_pairs.csv data/pairs/test_pairs.csv --cache_dir data/feature_cache --skip_hnr --max_workers 6
uv run python -m src.models.train --pairs_csv data/pairs/train_pairs.csv --val_csv data/pairs/val_pairs.csv --epochs 50 --batch_size 16 --lr 1e-4 --accumulation_steps 4 --output_dir models --feature_cache_dir data/feature_cache
uv run python -m src.models.train --eval_only --pairs_csv data/pairs/test_pairs.csv --model_path models/best_model.pt --feature_cache_dir data/feature_cache --batch_size 16
uv run python -m src.models.evaluate --model_path models/best_model.pt --pairs_csv data/pairs/test_pairs.csv --feature_cache_dir data/feature_cache
```

---

## Full Pipeline Refresh Run (CPU-only / Corporate Laptop)

Use this section when: datasets are already downloaded; you want to rebuild
pairs + feature cache from scratch and run all phases end-to-end on a machine
without a GPU (e.g. corporate laptop with Zscaler proxy).

**Constraints applied:**
- `--device cpu` — no CUDA required
- `--max_workers 4` — 6–8 core machine, 2–4 cores reserved for OS
- `--no-embeddings --no-lyrics` on inference — MERT/CLAP blocked by corporate proxy
- Tuning: 10 trials × 10 epochs (50 trials on CPU takes 12+ hours)

### Step 0 — Validate existing audio

```bash
uv run python data/validate_datasets.py --data_dir data
# If any invalid files are found:
uv run python data/validate_datasets.py --data_dir data --fix
# Then re-run the Phase 1 download command for the affected dataset
```

### Step 1 — Reconstruct training pairs

```bash
uv run python -m src.models.construct_pairs \
    --data_dir data \
    --output_dir data/pairs
```

### Step 2 — Precompute features with augmented variants

> With the optimised augmentation pipeline and `--skip_hnr`, expect
> **~10 minutes** with 6 workers. Only training files get augmented
> variants; val/test are cached raw-only. The script is **resumable**.

```bash
uv run python data/precompute_features.py \
    --pairs_csv data/pairs/train_pairs.csv \
                data/pairs/val_pairs.csv \
                data/pairs/test_pairs.csv \
    --cache_dir data/feature_cache \
    --skip_hnr \
    --max_workers 6 \
    --n_augmentations 5
```

### Step 3 — Train (quick-test: 10 epochs)

```bash
uv run python -m src.models.train \
    --pairs_csv data/pairs/train_pairs.csv \
    --val_csv data/pairs/val_pairs.csv \
    --epochs 10 \
    --batch_size 32 \
    --lr 1e-4 \
    --accumulation_steps 2 \
    --output_dir models \
    --feature_cache_dir data/feature_cache \
    --n_augmentations 5 \
    --no-augment \
    --device cpu
```

### Step 4 — Evaluate on test set

```bash
uv run python -m src.models.train \
    --eval_only \
    --pairs_csv data/pairs/test_pairs.csv \
    --model_path models/best_model.pt \
    --feature_cache_dir data/feature_cache \
    --batch_size 16 \
    --device cpu
```

### Step 5 — Hyperparameter tuning (10 trials)

> Tuning is resumable: re-run the same command to continue from `tune_study.db`.

```bash
uv run python -m src.models.tune \
    --pairs_csv data/pairs/train_pairs.csv \
    --val_csv data/pairs/val_pairs.csv \
    --n_trials 10 \
    --tuning_epochs 10 \
    --patience 5 \
    --feature_cache_dir data/feature_cache \
    --output_dir models/tune_checkpoints \
    --study_db models/tune_study.db
```

### Step 6 — Self-supervised pretraining

```bash
uv run python -m src.models.pretrain \
    --data_dir data \
    --output_dir models \
    --epochs 20 \
    --batch_size 32 \
    --lr 1e-4 \
    --aug_prob 0.5 \
    --device cpu
```

### Step 7 — Retrain with pretrained encoder *(depends on Step 6)*

```bash
uv run python -m src.models.train \
    --pairs_csv data/pairs/train_pairs.csv \
    --val_csv data/pairs/val_pairs.csv \
    --epochs 10 \
    --batch_size 32 \
    --lr 1e-4 \
    --accumulation_steps 2 \
    --output_dir models \
    --feature_cache_dir data/feature_cache \
    --n_augmentations 5 \
    --no-augment \
    --device cpu \
    --load-pretrained models/pretrained_best.pt
```

### Step 8 — Knowledge distillation *(depends on Step 3 or 7)*

```bash
uv run python -m src.models.distill \
    --teacher_path models/best_model.pt \
    --data_dir data \
    --output_dir models \
    --epochs 30 \
    --batch_size 32 \
    --student_dim 128 \
    --device cpu
```

### Step 9 — Final evaluation (metrics + plots)

```bash
uv run python -m src.models.train \
    --eval_only \
    --pairs_csv data/pairs/test_pairs.csv \
    --model_path models/best_model.pt \
    --feature_cache_dir data/feature_cache \
    --batch_size 16 \
    --device cpu

uv run python -m src.models.evaluate \
    --model_path models/best_model.pt \
    --pairs_csv data/pairs/test_pairs.csv \
    --feature_cache_dir data/feature_cache \
    --output_dir notebooks/figures \
    --device cpu
```

### Step 10 — Inference *(embeddings disabled — proxy blocks HuggingFace)*

```bash
# Real vs AI-generated (SONICS dataset)
uv run python -m src.compare_tracks \
    "data/sonics/audio/real_-8xdDaRFdwc.wav" \
    "data/sonics/audio/fake_53654_suno_1.wav" \
    --model models/best_model.pt \
    --no-embeddings \
    --no-lyrics \
    --mode standard \
    --json

# Two AI models compared (FakeMusicCaps)
uv run python -m src.compare_tracks \
    "data/fakemusiccaps/audio/musicldm_09lQmg2wvsY.wav" \
    "data/fakemusiccaps/audio/MusicGen_medium_09lQmg2wvsY.wav" \
    --model models/best_model.pt \
    --no-embeddings \
    --no-lyrics \
    --mode standard \
    --json
```

### Expected outputs after a full refresh

| Output | Step |
|---|---|
| `data/pairs/{train,val,test}_pairs.csv` | 1 |
| `data/feature_cache/*.npy` (~9600 files) | 2 |
| `models/best_model.pt` | 3 / 7 |
| `models/tune_checkpoints/trial_N/best_model.pt` | 5 |
| `models/tune_study.db` | 5 |
| `models/pretrained_best.pt` | 6 |
| `models/student_best.pt` | 8 |
| `mlruns/` (MLflow logs) | 3, 5, 7 |

---

## Project Structure

```
src/
  compare_tracks.py          # CLI entry point for pairwise comparison
  features/
    audio_features.py        # Tier 1 classical feature extraction
    augmentations.py         # Data augmentation (pitch shift, noise, codec)
    embedding_features.py    # MERT + CLAP deep embedding extraction
    speech_features.py       # Speech/vocal feature analysis
    lyrics_features.py       # Lyrical similarity via Whisper transcription
    source_separation.py     # Demucs vocal/instrument separation
    gpu_features.py          # GPU-accelerated feature extraction
    streaming.py             # Streaming audio processing for large files
  models/
    siamese_network.py       # Siamese network with attention pooling
    spectrogram_encoder.py   # CNN encoder for mel-spectrograms
    similarity_head.py       # Learned similarity scoring head
    pair_dataset.py          # PyTorch Dataset for audio pairs
    construct_pairs.py       # Pair construction from all datasets
    train.py                 # Training loop + evaluation
    evaluate.py              # Standalone evaluation with plots (ROC, confusion matrix)
    pretrain.py              # Self-supervised pretraining
    distill.py               # Knowledge distillation
    tune.py                  # Optuna hyperparameter tuning
    chunking.py              # Variable-length audio chunking
data/
  download_fakemusiccaps.py  # FakeMusicCaps downloader (Zenodo)
  download_mippia.py         # MIPPIA/SMP downloader (YouTube)
  download_sonics.py         # SONICS downloader (Kaggle)
  precompute_features.py     # Batch feature precomputation
notebooks/
  eda_fakemusiccaps.ipynb    # EDA: FakeMusicCaps dataset
  eda_mippia.ipynb           # EDA: MIPPIA dataset
  eda_sonics.ipynb           # EDA: SONICS dataset
  eda_cross_dataset.ipynb    # EDA: cross-dataset comparisons
tests/
  test_core.py               # Unit tests for core functionality
  test_downloads.py          # Tests for download scripts
  test_integration.py        # Integration tests
```
