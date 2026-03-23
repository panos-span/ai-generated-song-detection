"""Generate the four EDA Jupyter notebooks from source definitions.

Run this script whenever you want to regenerate the .ipynb files:
    uv run python notebooks/create_notebooks.py

Uses stdlib json only — no extra dependencies required.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cid() -> str:
    return uuid.uuid4().hex[:8]


def _code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": _cid(),
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def _md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": _cid(),
        "metadata": {},
        "source": source,
    }

# Aliases so the rest of the file is unchanged
new_code_cell = _code
new_markdown_cell = _md


def _nb(cells: list) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.13.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _save(nb: dict, name: str) -> None:
    path = NOTEBOOKS_DIR / name
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Shared snippet strings
# ---------------------------------------------------------------------------

_SHARED_IMPORTS = """\
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path("..").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.audio_features import (
    extract_tier1_features,
    load_audio,
)

sns.set_theme(style="whitegrid", palette="muted")
SR = 16000
FIGURES_DIR = PROJECT_ROOT / "notebooks" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
"""

_PROJECTION_HELPER = """\
def _reduce_2d(X: np.ndarray, n_samples_max: int = 200) -> np.ndarray:
    \"\"\"Reduce X to 2-D using UMAP (preferred) or t-SNE fallback.\"\"\"
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(X[:n_samples_max])
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42, verbose=False)
        print("  Using UMAP for projection")
    except ImportError:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X) - 1))
        print("  umap-learn not installed, falling back to t-SNE")
    return reducer.fit_transform(X)
"""

_EMBEDDING_EXTRACTION = """\
# ── Extract MERT + CLAP embeddings (≤30 files per label; increase as needed) ──
from src.features.embedding_features import EmbeddingExtractor

MAX_PER_CLASS = 30   # increase if you have a GPU and more time

extractor = EmbeddingExtractor(device="auto", use_mert=True, use_clap=True)
"""


# ---------------------------------------------------------------------------
# Notebook 1: SONICS
# ---------------------------------------------------------------------------

def _sonics_cells() -> list:
    return [
        new_markdown_cell("# EDA — SONICS Dataset\n\nExploratory analysis of the SONICS dataset: binary real/AI classification with Suno and Udio as generators."),
        new_code_cell(_SHARED_IMPORTS),
        new_code_cell("""\
# ── Load metadata ──
meta = pd.read_csv(PROJECT_ROOT / "data" / "sonics" / "metadata.csv")
print(f"Shape: {meta.shape}")
display(meta.head())
display(meta.describe(include="all"))
"""),
        new_code_cell("""\
# ── Label distribution ──
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

sns.countplot(data=meta, x="label", palette="Set2", ax=axes[0])
axes[0].set_title("Label distribution")
axes[0].bar_label(axes[0].containers[0])

label_counts = meta["label"].value_counts()
axes[1].pie(label_counts, labels=label_counts.index, autopct="%1.1f%%",
            colors=sns.color_palette("Set2", len(label_counts)))
axes[1].set_title("Label share")

plt.tight_layout()
fig.savefig(FIGURES_DIR / "sonics_label_distribution.png", dpi=150)
plt.show()
"""),
        new_code_cell("""\
# ── Generator distribution ──
fig, ax = plt.subplots(figsize=(7, 4))
order = meta["generator"].value_counts().index
sns.countplot(data=meta, x="generator", order=order, palette="tab10", ax=ax)
ax.set_title("Generator distribution")
ax.bar_label(ax.containers[0])
plt.tight_layout()
fig.savefig(FIGURES_DIR / "sonics_generator_distribution.png", dpi=150)
plt.show()
"""),
        new_code_cell("""\
# ── Split distribution stacked by label ──
if "split" in meta.columns:
    split_label = meta.groupby(["split", "label"]).size().unstack(fill_value=0)
    split_label.plot(kind="bar", stacked=True, colormap="Set2", figsize=(7, 4))
    plt.title("Train/val/test split by label")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sonics_split_distribution.png", dpi=150)
    plt.show()
"""),
        new_code_cell("""\
# ── Duration distribution by label ──
if "duration" in meta.columns:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(data=meta, x="duration", hue="label", bins=30, kde=True,
                 palette="Set2", ax=ax)
    ax.set_title("Duration distribution by label")
    ax.set_xlabel("Duration (s)")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "sonics_duration_distribution.png", dpi=150)
    plt.show()
else:
    print("No 'duration' column in metadata — skipping duration plot.")
"""),
        new_code_cell("""\
# ── Waveform + mel-spectrogram for one track per label ──
import librosa
import librosa.display

audio_dir = PROJECT_ROOT / "data" / "sonics" / "audio"
labels = meta["label"].unique()

fig, axes = plt.subplots(len(labels), 2, figsize=(14, 4 * len(labels)))
if len(labels) == 1:
    axes = [axes]

for row_idx, lbl in enumerate(sorted(labels)):
    row = meta[meta["label"] == lbl].iloc[0]
    path = str(audio_dir / row["filename"])
    audio, _ = load_audio(path, sr=SR)
    duration = len(audio) / SR

    t = np.linspace(0, duration, len(audio))
    axes[row_idx][0].plot(t, audio, linewidth=0.4)
    axes[row_idx][0].set_title(f"[{lbl}] Waveform — {row['filename']}")
    axes[row_idx][0].set_xlabel("Time (s)")

    mel = librosa.feature.melspectrogram(y=audio, sr=SR, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    librosa.display.specshow(mel_db, sr=SR, hop_length=512,
                             x_axis="time", y_axis="mel", ax=axes[row_idx][1])
    axes[row_idx][1].set_title(f"[{lbl}] Mel-spectrogram")

plt.tight_layout()
fig.savefig(FIGURES_DIR / "sonics_spectrograms.png", dpi=150)
plt.show()
"""),
        new_code_cell("""\
# ── Tier 1 + Tier 2 feature box-plots (top-10 most discriminative dims) ──
from src.features.audio_features import extract_all_features

N_PER_CLASS = 20  # increase for more robust statistics

feat_dict: dict[str, dict[str, np.ndarray]] = {}
for lbl in meta["label"].unique():
    rows = meta[meta["label"] == lbl].head(N_PER_CLASS)
    tier1_list, tier2_list = [], []
    for _, r in rows.iterrows():
        p = str(audio_dir / r["filename"])
        a, _ = load_audio(p, sr=SR)
        f = extract_all_features(a, SR)
        tier1_list.append(f["tier1"])
        tier2_list.append(f["tier2"])
    feat_dict[lbl] = {
        "tier1": np.array(tier1_list),
        "tier2": np.array(tier2_list),
    }

def _top_discriminative(matrices: dict[str, np.ndarray], top_k: int = 10) -> list[int]:
    keys = list(matrices.keys())
    a, b = matrices[keys[0]], matrices[keys[1]]
    pooled_std = np.sqrt((np.var(a, axis=0) + np.var(b, axis=0)) / 2 + 1e-10)
    d = np.abs(a.mean(axis=0) - b.mean(axis=0)) / pooled_std
    return list(np.argsort(d)[::-1][:top_k])

for tier_name in ("tier1", "tier2"):
    mats = {k: v[tier_name] for k, v in feat_dict.items()}
    top_idx = _top_discriminative(mats)

    records = []
    for lbl, mat in mats.items():
        for i, fi in enumerate(top_idx):
            for val in mat[:, fi]:
                records.append({"label": lbl, "feature_idx": f"f{fi}", "value": float(val)})
    df_box = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(14, 4))
    sns.boxplot(data=df_box, x="feature_idx", y="value", hue="label",
                palette="Set2", ax=ax)
    ax.set_title(f"SONICS — {tier_name.upper()} top-10 discriminative features")
    ax.set_xlabel("Feature index")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / f"sonics_{tier_name}_boxplot.png", dpi=150)
    plt.show()
"""),
        new_code_cell(_PROJECTION_HELPER),
        new_code_cell(_EMBEDDING_EXTRACTION + """\

emb_records: list[dict] = []
for lbl in meta["label"].unique():
    rows = meta[meta["label"] == lbl].head(MAX_PER_CLASS)
    for _, r in rows.iterrows():
        p = str(audio_dir / r["filename"])
        audio, _ = load_audio(p, sr=SR)
        embs = extractor.extract_all_embeddings(audio, SR)
        emb_records.append({
            "label": lbl,
            "generator": r.get("generator", lbl),
            **{k: v for k, v in embs.items()},
        })

print(f"Extracted embeddings for {len(emb_records)} tracks")
"""),
        new_code_cell("""\
# ── t-SNE / UMAP scatter ──
for emb_key in ("mert", "clap"):
    vecs = [r[emb_key] for r in emb_records if emb_key in r]
    labels_plot = [r["label"] for r in emb_records if emb_key in r]
    generators_plot = [r["generator"] for r in emb_records if emb_key in r]

    if len(vecs) < 4:
        print(f"Not enough samples for {emb_key} projection")
        continue

    coords = _reduce_2d(np.array(vecs))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    palette = sns.color_palette("Set2", n_colors=len(set(labels_plot)))
    for ax, hue, title in [
        (axes[0], labels_plot, "by label"),
        (axes[1], generators_plot, "by generator"),
    ]:
        unique = sorted(set(hue))
        pal = sns.color_palette("tab10", len(unique))
        color_map = dict(zip(unique, pal))
        for grp in unique:
            idx = [i for i, h in enumerate(hue) if h == grp]
            ax.scatter(coords[idx, 0], coords[idx, 1], label=grp,
                       color=color_map[grp], alpha=0.75, s=40)
        ax.legend(fontsize=8)
        ax.set_title(f"SONICS {emb_key.upper()} 2-D projection — {title}")
        ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / f"sonics_{emb_key}_projection.png", dpi=150)
    plt.show()
"""),
    ]


# ---------------------------------------------------------------------------
# Notebook 2: FakeMusicCaps
# ---------------------------------------------------------------------------

def _fakemusiccaps_cells() -> list:
    return [
        new_markdown_cell("# EDA — FakeMusicCaps Dataset\n\nExploratory analysis of FakeMusicCaps: AI-generated music from 5 different generative models, paired with MusicCaps text captions."),
        new_code_cell(_SHARED_IMPORTS),
        new_code_cell("""\
# ── Load metadata ──
meta = pd.read_csv(PROJECT_ROOT / "data" / "fakemusiccaps" / "metadata.csv")
print(f"Shape: {meta.shape}")
display(meta.head())
display(meta.describe(include="all"))
print("\\nNull counts:\\n", meta.isnull().sum())
"""),
        new_code_cell("""\
# ── Model distribution ──
fig, ax = plt.subplots(figsize=(9, 4))
order = meta["model"].value_counts().index
sns.countplot(data=meta, x="model", order=order, palette="tab10", ax=ax)
ax.set_title("FakeMusicCaps — tracks per generative model")
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
ax.bar_label(ax.containers[0])
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fakemusiccaps_model_distribution.png", dpi=150)
plt.show()
"""),
        new_code_cell("""\
# ── Caption analysis ──
if "caption" in meta.columns:
    meta["caption_len"] = meta["caption"].fillna("").str.split().str.len()
    has_caption = (meta["caption"].notna() & (meta["caption"].str.strip() != ""))
    print(f"Tracks with captions: {has_caption.sum()} / {len(meta)} "
          f"({100 * has_caption.mean():.1f}%)")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(data=meta[has_caption], x="caption_len", bins=30, kde=True, ax=ax)
    ax.set_title("Caption length distribution (word count)")
    ax.set_xlabel("Words")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fakemusiccaps_caption_lengths.png", dpi=150)
    plt.show()

    # Top-20 words
    from collections import Counter
    all_words = " ".join(meta["caption"].dropna()).lower().split()
    stopwords = {"a","the","and","of","with","in","on","is","to","it","an","for","are","by"}
    counter = Counter(w for w in all_words if w not in stopwords and len(w) > 2)
    top20 = pd.DataFrame(counter.most_common(20), columns=["word", "count"])
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=top20, x="word", y="count", palette="Blues_d", ax=ax)
    ax.set_title("Top-20 caption words")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fakemusiccaps_caption_words.png", dpi=150)
    plt.show()
else:
    print("No 'caption' column found.")
"""),
        new_code_cell("""\
# ── Duration distribution per model ──
if "duration" in meta.columns:
    g = sns.FacetGrid(meta, col="model", col_wrap=3, height=3, sharey=False)
    g.map(sns.histplot, "duration", bins=20, kde=True, color="#4C9BE8")
    g.set_titles("{col_name}")
    g.set_axis_labels("Duration (s)", "Count")
    g.figure.suptitle("Duration distribution per generator", y=1.02)
    plt.tight_layout()
    g.figure.savefig(FIGURES_DIR / "fakemusiccaps_duration_by_model.png", dpi=150)
    plt.show()
else:
    print("No 'duration' column — skipping.")
"""),
        new_code_cell("""\
# ── One waveform + mel-spec per model ──
import librosa, librosa.display
audio_dir = PROJECT_ROOT / "data" / "fakemusiccaps" / "audio"
models = sorted(meta["model"].unique())

fig, axes = plt.subplots(len(models), 2, figsize=(14, 4 * len(models)))
for row_idx, model in enumerate(models):
    row = meta[meta["model"] == model].iloc[0]
    path = str(audio_dir / row["filename"])
    audio, _ = load_audio(path, sr=SR)
    t = np.linspace(0, len(audio) / SR, len(audio))

    axes[row_idx][0].plot(t, audio, linewidth=0.3)
    axes[row_idx][0].set_title(f"[{model}] Waveform")
    axes[row_idx][0].set_xlabel("Time (s)")

    mel = librosa.feature.melspectrogram(y=audio, sr=SR, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    librosa.display.specshow(mel_db, sr=SR, hop_length=512,
                             x_axis="time", y_axis="mel", ax=axes[row_idx][1])
    axes[row_idx][1].set_title(f"[{model}] Mel-spec")

plt.tight_layout()
fig.savefig(FIGURES_DIR / "fakemusiccaps_spectrograms.png", dpi=150)
plt.show()
"""),
        new_code_cell("""\
# ── Tier 1 mean-per-model heatmap (top-20 dims by variance) ──
N_PER_MODEL = 15  # increase for better stats
audio_dir = PROJECT_ROOT / "data" / "fakemusiccaps" / "audio"
models = sorted(meta["model"].unique())

model_means: dict[str, np.ndarray] = {}
for model in models:
    rows = meta[meta["model"] == model].head(N_PER_MODEL)
    feats = []
    for _, r in rows.iterrows():
        a, _ = load_audio(str(audio_dir / r["filename"]), sr=SR)
        feats.append(extract_tier1_features(a, SR))
    arr = np.array(feats)
    model_means[model] = arr.mean(axis=0)

mean_mat = np.array(list(model_means.values()))   # (n_models, n_features)

# Keep top-20 dims by inter-model variance
top20_idx = np.argsort(mean_mat.var(axis=0))[::-1][:20]
mean_sub = mean_mat[:, top20_idx]

from sklearn.preprocessing import StandardScaler
mean_sub_z = StandardScaler().fit_transform(mean_sub.T).T  # z-score across models

df_heat = pd.DataFrame(mean_sub_z, index=models,
                       columns=[f"f{i}" for i in top20_idx])
fig, ax = plt.subplots(figsize=(14, 4))
sns.heatmap(df_heat, cmap="RdBu_r", center=0, ax=ax, linewidths=0.3,
            cbar_kws={"label": "z-score"})
ax.set_title("FakeMusicCaps — Tier 1 mean features per model (top-20 by variance, z-scored)")
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fakemusiccaps_tier1_heatmap.png", dpi=150)
plt.show()
"""),
        new_code_cell(_PROJECTION_HELPER),
        new_code_cell(_EMBEDDING_EXTRACTION + """\
audio_dir = PROJECT_ROOT / "data" / "fakemusiccaps" / "audio"

emb_records: list[dict] = []
for model in meta["model"].unique():
    rows = meta[meta["model"] == model].head(MAX_PER_CLASS)
    for _, r in rows.iterrows():
        p = str(audio_dir / r["filename"])
        audio, _ = load_audio(p, sr=SR)
        embs = extractor.extract_all_embeddings(audio, SR)
        emb_records.append({"model": model, **{k: v for k, v in embs.items()}})

print(f"Extracted embeddings for {len(emb_records)} tracks")
"""),
        new_code_cell("""\
# ── t-SNE / UMAP scatter coloured by model ──
for emb_key in ("mert", "clap"):
    vecs = [r[emb_key] for r in emb_records if emb_key in r]
    model_labels = [r["model"] for r in emb_records if emb_key in r]
    if len(vecs) < 4:
        continue

    coords = _reduce_2d(np.array(vecs))
    unique_models = sorted(set(model_labels))
    pal = sns.color_palette("tab10", len(unique_models))
    color_map = dict(zip(unique_models, pal))

    fig, ax = plt.subplots(figsize=(8, 6))
    for m in unique_models:
        idx = [i for i, ml in enumerate(model_labels) if ml == m]
        ax.scatter(coords[idx, 0], coords[idx, 1], label=m,
                   color=color_map[m], alpha=0.75, s=45)
    ax.legend(fontsize=8, loc="best")
    ax.set_title(f"FakeMusicCaps {emb_key.upper()} 2-D projection (by model)")
    ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / f"fakemusiccaps_{emb_key}_projection.png", dpi=150)
    plt.show()
"""),
    ]


# ---------------------------------------------------------------------------
# Notebook 3: MIPPIA
# ---------------------------------------------------------------------------

def _mippia_cells() -> list:
    return [
        new_markdown_cell("# EDA — MIPPIA / SMP Dataset\n\nExploratory analysis of the MIPPIA plagiarism dataset: 70 song pairs with relation labels (`plag`, `plag_doubt`, `remake`) and segment-level timestamps."),
        new_code_cell(_SHARED_IMPORTS),
        new_code_cell("""\
# ── Load metadata ──
meta = pd.read_csv(PROJECT_ROOT / "data" / "mippia" / "metadata.csv")
segments = pd.read_csv(PROJECT_ROOT / "data" / "mippia" / "smp_dataset" / "Final_dataset_pairs.csv")

print("Pair-level metadata shape:", meta.shape)
display(meta.head())
print("\\nSegment-level CSV shape:", segments.shape)
display(segments.head())
"""),
        new_code_cell("""\
# ── Relation type distribution ──
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

order = meta["similarity_label"].value_counts().index
sns.countplot(data=meta, x="similarity_label", order=order, palette="Set2", ax=axes[0])
axes[0].set_title("Pair relation type distribution")
axes[0].bar_label(axes[0].containers[0])

pct = (meta["similarity_label"].value_counts() / len(meta) * 100).round(1)
print("Relation type percentages:")
print(pct.to_string())
pct.plot.pie(autopct="%1.1f%%", ax=axes[1],
             colors=sns.color_palette("Set2", len(pct)))
axes[1].set_ylabel("")
axes[1].set_title("Share by relation type")

plt.tight_layout()
fig.savefig(FIGURES_DIR / "mippia_relation_distribution.png", dpi=150)
plt.show()
"""),
        new_code_cell("""\
# ── Segment temporal analysis ──
import ast

def _parse_times(s):
    try:
        return ast.literal_eval(str(s)) if pd.notna(s) else []
    except Exception:
        return []

segments["ori_times_list"] = segments["ori_times"].apply(_parse_times)
segments["comp_times_list"] = segments["comp_times"].apply(_parse_times)

# Flatten: each pair of timestamps is [start, end]
ori_starts = [t[0] for ts in segments["ori_times_list"] for t in (ts if isinstance(ts[0], list) else [ts]) if len(t) >= 2]
ori_ends   = [t[1] for ts in segments["ori_times_list"] for t in (ts if isinstance(ts[0], list) else [ts]) if len(t) >= 2]
seg_durations = [e - s for s, e in zip(ori_starts, ori_ends) if e > s]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(ori_starts, bins=20, color="#4C9BE8", edgecolor="black")
axes[0].set_title("Similar segment start times (original)")
axes[0].set_xlabel("Time (s)")

if seg_durations:
    axes[1].hist(seg_durations, bins=20, color="#F48024", edgecolor="black")
    axes[1].set_title("Similar segment durations")
    axes[1].set_xlabel("Duration (s)")
else:
    axes[1].text(0.5, 0.5, "Could not parse segment durations",
                 ha="center", va="center", transform=axes[1].transAxes)

plt.tight_layout()
fig.savefig(FIGURES_DIR / "mippia_segment_analysis.png", dpi=150)
plt.show()
"""),
        new_code_cell("""\
# ── Waveform side-by-side for 3 example pairs ──
import librosa, librosa.display

audio_dir = PROJECT_ROOT / "data" / "mippia" / "audio"
example_pairs = meta.head(3)

fig, axes = plt.subplots(len(example_pairs), 2, figsize=(14, 3 * len(example_pairs)))
for row_idx, (_, pair) in enumerate(example_pairs.iterrows()):
    for col_idx, track_col in enumerate(("track_a", "track_b")):
        fpath = audio_dir / pair[track_col]
        if not fpath.exists():
            axes[row_idx][col_idx].text(0.5, 0.5, f"File not found:\\n{pair[track_col]}",
                                        ha="center", va="center",
                                        transform=axes[row_idx][col_idx].transAxes)
            axes[row_idx][col_idx].set_title(f"Pair {pair['pair_id']} — {track_col}")
            continue
        audio, _ = load_audio(str(fpath), sr=SR)
        t = np.linspace(0, len(audio) / SR, len(audio))
        axes[row_idx][col_idx].plot(t, audio, linewidth=0.3)
        track_label = "Original" if track_col == "track_a" else "Comparison"
        axes[row_idx][col_idx].set_title(
            f"Pair {pair['pair_id']} ({pair['similarity_label']}) — {track_label}"
        )
        axes[row_idx][col_idx].set_xlabel("Time (s)")

plt.tight_layout()
fig.savefig(FIGURES_DIR / "mippia_example_waveforms.png", dpi=150)
plt.show()
"""),
        new_code_cell("""\
# ── Compare tracks: similarity score per relation type ──
# Runs compare_tracks() in fast mode (Fourier only) for up to 20 pairs.
from src.compare_tracks import compare_tracks

MAX_PAIRS = 20
score_records = []
for _, pair in meta.head(MAX_PAIRS).iterrows():
    path_a = str(audio_dir / pair["track_a"])
    path_b = str(audio_dir / pair["track_b"])
    if not Path(path_a).exists() or not Path(path_b).exists():
        continue
    try:
        result = compare_tracks(path_a, path_b, mode="fast",
                                use_embeddings=False, use_lyrics=False, use_vocals=False)
        score_records.append({
            "pair_id": pair["pair_id"],
            "similarity_label": pair["similarity_label"],
            "attribution_score": result["attribution_score"],
        })
    except Exception as e:
        print(f"  Pair {pair['pair_id']} failed: {e}")

df_scores = pd.DataFrame(score_records)
print(df_scores)

fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(data=df_scores, x="similarity_label", y="attribution_score",
            order=["plag", "plag_doubt", "remake"], palette="Set2", ax=ax)
ax.set_title("Attribution score by relation type (fast mode, Fourier)")
ax.set_ylabel("Attribution score")
plt.tight_layout()
fig.savefig(FIGURES_DIR / "mippia_scores_by_relation.png", dpi=150)
plt.show()
"""),
        new_code_cell("""\
# ── Tier 1 cosine similarity between track_a / track_b per relation ──
from scipy.spatial.distance import cosine as cosine_distance

MAX_PAIRS_FEATURES = 20
cos_records = []
for _, pair in meta.head(MAX_PAIRS_FEATURES).iterrows():
    path_a = str(audio_dir / pair["track_a"])
    path_b = str(audio_dir / pair["track_b"])
    if not Path(path_a).exists() or not Path(path_b).exists():
        continue
    a_audio, _ = load_audio(path_a, sr=SR)
    b_audio, _ = load_audio(path_b, sr=SR)
    fa = extract_tier1_features(a_audio, SR)
    fb = extract_tier1_features(b_audio, SR)
    cos_sim = float(1.0 - cosine_distance(fa, fb))
    cos_records.append({"similarity_label": pair["similarity_label"], "cosine_sim": cos_sim})

df_cos = pd.DataFrame(cos_records)
fig, ax = plt.subplots(figsize=(7, 4))
sns.boxplot(data=df_cos, x="similarity_label", y="cosine_sim",
            order=["plag", "plag_doubt", "remake"], palette="Set2", ax=ax)
ax.set_title("Tier 1 cosine similarity by relation type")
ax.set_ylabel("Cosine similarity")
plt.tight_layout()
fig.savefig(FIGURES_DIR / "mippia_tier1_cosine_by_relation.png", dpi=150)
plt.show()
"""),
        new_code_cell(_PROJECTION_HELPER),
        new_code_cell(_EMBEDDING_EXTRACTION + """\
audio_dir = PROJECT_ROOT / "data" / "mippia" / "audio"

emb_records: list[dict] = []
for _, pair in meta.iterrows():
    for track_col in ("track_a", "track_b"):
        fpath = audio_dir / pair[track_col]
        if not fpath.exists():
            continue
        audio, _ = load_audio(str(fpath), sr=SR)
        embs = extractor.extract_all_embeddings(audio, SR)
        emb_records.append({
            "similarity_label": pair["similarity_label"],
            "track": track_col,
            **{k: v for k, v in embs.items()},
        })
        if len(emb_records) >= MAX_PER_CLASS * 2:
            break
    if len(emb_records) >= MAX_PER_CLASS * 2:
        break

print(f"Extracted embeddings for {len(emb_records)} tracks")
"""),
        new_code_cell("""\
# ── t-SNE / UMAP scatter coloured by relation type ──
for emb_key in ("mert", "clap"):
    vecs = [r[emb_key] for r in emb_records if emb_key in r]
    rel_labels = [r["similarity_label"] for r in emb_records if emb_key in r]
    if len(vecs) < 4:
        continue

    coords = _reduce_2d(np.array(vecs))
    unique_rels = sorted(set(rel_labels))
    pal = sns.color_palette("Set2", len(unique_rels))
    color_map = dict(zip(unique_rels, pal))

    fig, ax = plt.subplots(figsize=(8, 6))
    for rel in unique_rels:
        idx = [i for i, l in enumerate(rel_labels) if l == rel]
        ax.scatter(coords[idx, 0], coords[idx, 1], label=rel,
                   color=color_map[rel], alpha=0.75, s=45)
    ax.legend(fontsize=9)
    ax.set_title(f"MIPPIA {emb_key.upper()} 2-D projection (by relation type)")
    ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / f"mippia_{emb_key}_projection.png", dpi=150)
    plt.show()
"""),
    ]


# ---------------------------------------------------------------------------
# Notebook 4: Cross-dataset comparison
# ---------------------------------------------------------------------------

def _cross_dataset_cells() -> list:
    return [
        new_markdown_cell("# EDA — Cross-Dataset Comparison\n\nCompares SONICS, FakeMusicCaps, and MIPPIA across sample counts, duration distributions, feature separability, and deep embedding projections."),
        new_code_cell(_SHARED_IMPORTS),
        new_code_cell("""\
# ── Load and merge all three metadata CSVs ──
def _load_meta(dataset: str, label_col: str | None = None) -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / dataset / "metadata.csv"
    df = pd.read_csv(path)
    df["source"] = dataset
    if label_col and label_col in df.columns:
        df["class_label"] = df[label_col].astype(str)
    elif "model" in df.columns:
        df["class_label"] = df["model"].astype(str)
    elif "similarity_label" in df.columns:
        df["class_label"] = df["similarity_label"].astype(str)
    else:
        df["class_label"] = dataset
    return df

sonics = _load_meta("sonics", label_col="label")
if "generator" in sonics.columns:
    sonics["class_label"] = sonics["generator"]

fakemusiccaps = _load_meta("fakemusiccaps", label_col="model")
mippia = _load_meta("mippia", label_col="similarity_label")

# Use a common audio filename column
for df, col in [(sonics, "filename"), (fakemusiccaps, "filename"),
                (mippia, "track_a")]:
    if col in df.columns and "filename" not in df.columns:
        df["filename"] = df[col]

combined = pd.concat([sonics, fakemusiccaps, mippia], ignore_index=True)
print(f"Combined shape: {combined.shape}")
display(combined[["source", "class_label", "filename"]].head(10))
print("\\nRows per dataset:", combined["source"].value_counts().to_dict())
"""),
        new_code_cell("""\
# ── Sample counts by dataset × class ──
fig, ax = plt.subplots(figsize=(12, 5))
counts = combined.groupby(["source", "class_label"]).size().reset_index(name="count")
sns.barplot(data=counts, x="class_label", y="count", hue="source",
            palette="tab10", ax=ax)
ax.set_title("Sample count by dataset and class")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
plt.tight_layout()
fig.savefig(FIGURES_DIR / "cross_sample_counts.png", dpi=150)
plt.show()
"""),
        new_code_cell("""\
# ── Duration violin plot by dataset ──
if "duration" in combined.columns:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.violinplot(data=combined.dropna(subset=["duration"]),
                   x="source", y="duration", palette="Set2",
                   inner="quartile", ax=ax)
    ax.set_title("Duration distribution by dataset")
    ax.set_ylabel("Duration (s)")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "cross_duration_violin.png", dpi=150)
    plt.show()
else:
    print("No 'duration' column in merged metadata — skipping violin plot.")
"""),
        new_code_cell("""\
# ── PCA variance-explained per dataset ──
from sklearn.decomposition import PCA

N_TRACKS = 20   # per dataset class; increase for better estimates
audio_dirs = {
    "sonics": PROJECT_ROOT / "data" / "sonics" / "audio",
    "fakemusiccaps": PROJECT_ROOT / "data" / "fakemusiccaps" / "audio",
    "mippia": PROJECT_ROOT / "data" / "mippia" / "audio",
}
mippia_filename_col = "track_a"  # MIPPIA uses pair filenames

dataset_features: dict[str, np.ndarray] = {}
for ds in ["sonics", "fakemusiccaps", "mippia"]:
    sub = combined[combined["source"] == ds]
    fname_col = mippia_filename_col if ds == "mippia" else "filename"
    if fname_col not in sub.columns:
        fname_col = "filename"
    feats = []
    for _, r in sub.head(N_TRACKS).iterrows():
        try:
            p = str(audio_dirs[ds] / r[fname_col])
            a, _ = load_audio(p, sr=SR)
            feats.append(extract_tier1_features(a, SR))
        except Exception:
            continue
    if feats:
        dataset_features[ds] = np.array(feats)

fig, ax = plt.subplots(figsize=(8, 4))
palette = sns.color_palette("tab10", len(dataset_features))
for (ds, mat), color in zip(dataset_features.items(), palette):
    pca = PCA(n_components=min(mat.shape[0], mat.shape[1], 30))
    pca.fit(mat)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    ax.plot(range(1, len(cum_var) + 1), cum_var, label=ds, color=color, marker="o", ms=3)

ax.axhline(0.9, ls="--", color="grey", alpha=0.6, label="90% threshold")
ax.set_title("PCA cumulative variance explained per dataset (Tier 1)")
ax.set_xlabel("Number of components")
ax.set_ylabel("Cumulative explained variance")
ax.legend()
plt.tight_layout()
fig.savefig(FIGURES_DIR / "cross_pca_variance.png", dpi=150)
plt.show()
"""),
        new_code_cell("""\
# ── Cohen's d heatmap — inter-dataset feature separability ──
from itertools import combinations

all_ds = list(dataset_features.keys())
if len(all_ds) >= 2:
    pairs = list(combinations(all_ds, 2))
    max_dim = min(mat.shape[1] for mat in dataset_features.values())
    cohens_d_mat = np.zeros((len(pairs), max_dim))

    for i, (ds_a, ds_b) in enumerate(pairs):
        a = dataset_features[ds_a][:, :max_dim]
        b = dataset_features[ds_b][:, :max_dim]
        pooled_std = np.sqrt((np.var(a, axis=0) + np.var(b, axis=0)) / 2 + 1e-10)
        cohens_d_mat[i] = np.abs(a.mean(axis=0) - b.mean(axis=0)) / pooled_std

    # Show only top-30 most discriminative feature dimensions (across all pairs)
    top30 = np.argsort(cohens_d_mat.max(axis=0))[::-1][:30]
    df_heat = pd.DataFrame(
        cohens_d_mat[:, top30],
        index=[f"{a} vs {b}" for a, b in pairs],
        columns=[f"f{i}" for i in top30],
    )
    fig, ax = plt.subplots(figsize=(16, 3))
    sns.heatmap(df_heat, cmap="YlOrRd", ax=ax, linewidths=0.2,
                cbar_kws={"label": "Cohen's d"})
    ax.set_title("Inter-dataset Cohen's d heatmap (Tier 1 features, top-30 dims)")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "cross_cohens_d_heatmap.png", dpi=150)
    plt.show()
"""),
        new_code_cell(_PROJECTION_HELPER),
        new_code_cell(_EMBEDDING_EXTRACTION + """\
all_audio_dirs = {
    "sonics": PROJECT_ROOT / "data" / "sonics" / "audio",
    "fakemusiccaps": PROJECT_ROOT / "data" / "fakemusiccaps" / "audio",
    "mippia": PROJECT_ROOT / "data" / "mippia" / "audio",
}

emb_records: list[dict] = []
for ds in ["sonics", "fakemusiccaps", "mippia"]:
    sub = combined[combined["source"] == ds]
    fname_col = "track_a" if ds == "mippia" else "filename"
    if fname_col not in sub.columns:
        fname_col = "filename"
    for _, r in sub.head(MAX_PER_CLASS).iterrows():
        try:
            p = str(all_audio_dirs[ds] / r[fname_col])
            audio, _ = load_audio(p, sr=SR)
            embs = extractor.extract_all_embeddings(audio, SR)
            emb_records.append({
                "source": ds,
                "class_label": r["class_label"],
                **{k: v for k, v in embs.items()},
            })
        except Exception as e:
            pass

print(f"Extracted embeddings for {len(emb_records)} tracks across all datasets")
"""),
        new_code_cell("""\
# ── Combined t-SNE / UMAP scatter across all datasets ──
for emb_key in ("mert", "clap"):
    vecs = [r[emb_key] for r in emb_records if emb_key in r]
    sources = [r["source"] for r in emb_records if emb_key in r]
    classes = [r["class_label"] for r in emb_records if emb_key in r]
    if len(vecs) < 4:
        continue

    coords = _reduce_2d(np.array(vecs))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, hue, title in [(axes[0], sources, "by dataset"),
                           (axes[1], classes, "by class label")]:
        unique = sorted(set(hue))
        pal = sns.color_palette("tab20", len(unique))
        cmap = dict(zip(unique, pal))
        for grp in unique:
            idx = [i for i, h in enumerate(hue) if h == grp]
            ax.scatter(coords[idx, 0], coords[idx, 1], label=grp,
                       color=cmap[grp], alpha=0.7, s=35)
        ax.legend(fontsize=7, ncol=2)
        ax.set_title(f"All datasets {emb_key.upper()} — {title}")
        ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / f"cross_{emb_key}_projection.png", dpi=150)
    plt.show()
"""),
        new_code_cell("""\
# ── Summary table: most-separable Tier 1 features per dataset ──
print("\\n" + "=" * 60)
print("  CROSS-DATASET SUMMARY")
print("=" * 60)

for ds, mat in dataset_features.items():
    n_comp_90 = 0
    pca = PCA()
    pca.fit(mat)
    cum = np.cumsum(pca.explained_variance_ratio_)
    n_comp_90 = int(np.searchsorted(cum, 0.9) + 1)
    print(f"\\n{ds.upper()}  ({mat.shape[0]} tracks, {mat.shape[1]} Tier-1 dims)")
    print(f"  PCA components for 90% variance: {n_comp_90}")
    print(f"  Feature mean: {mat.mean():.4f}, std: {mat.std():.4f}")
"""),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Generating EDA notebooks...")

    _save(_nb(_sonics_cells()), "eda_sonics.ipynb")
    _save(_nb(_fakemusiccaps_cells()), "eda_fakemusiccaps.ipynb")
    _save(_nb(_mippia_cells()), "eda_mippia.ipynb")
    _save(_nb(_cross_dataset_cells()), "eda_cross_dataset.ipynb")

    print("Done. Run notebooks with:")
    print("  jupyter lab notebooks/")


if __name__ == "__main__":
    main()
