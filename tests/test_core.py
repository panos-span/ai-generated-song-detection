import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch

from src.features.audio_features import (
    extract_all_features,
    extract_tier1_features,
    extract_tier2_features,
)
from src.models.chunking import chunk_audio
from src.models.siamese_network import AttentionPooling, FeatureProjector, MultiHeadAttentionPooling, ProjectionHead, SiameseNetwork
from src.models.similarity_head import PairwiseSimilarityModel, SimilarityHead
from src.models.train import FocalLoss
from src.models.pair_dataset import compute_feature_stats
from src.compare_tracks import (
    compute_ai_artifact_score,
    embedding_similarity,
    melodic_similarity,
    structural_similarity,
    timbral_similarity,
)

SR = 16000


@pytest.fixture
def sine_3s():
    t = np.linspace(0, 3, 3 * SR, endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


@pytest.fixture
def white_noise_3s():
    rng = np.random.default_rng(42)
    return rng.standard_normal(3 * SR).astype(np.float32) * 0.1


@pytest.fixture
def silence_3s():
    return np.zeros(3 * SR, dtype=np.float32)


# -- Audio Features Tests --


class TestAudioFeatures:
    def test_tier1_feature_shape(self, sine_3s):
        feats = extract_tier1_features(sine_3s, SR)
        assert feats.shape == (430,)

    def test_tier2_feature_shape(self, sine_3s):
        feats = extract_tier2_features(sine_3s, SR)
        assert feats.shape == (22,)

    def test_all_features_keys(self, sine_3s):
        feats = extract_all_features(sine_3s, SR)
        assert set(feats.keys()) == {"tier1", "tier2", "combined"}

    def test_combined_feature_shape(self, sine_3s):
        feats = extract_all_features(sine_3s, SR)
        expected_len = feats["tier1"].shape[0] + feats["tier2"].shape[0]
        assert feats["combined"].shape == (expected_len,)

    def test_features_no_nan(self, sine_3s):
        feats = extract_all_features(sine_3s, SR)
        for key in ("tier1", "tier2", "combined"):
            assert not np.any(np.isnan(feats[key])), f"NaN found in {key}"
            assert not np.any(np.isinf(feats[key])), f"Inf found in {key}"

    def test_short_audio(self):
        short = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, int(0.1 * SR), endpoint=False)).astype(
            np.float32
        )
        t1 = extract_tier1_features(short, SR)
        t2 = extract_tier2_features(short, SR)
        assert t1.shape == (430,)
        assert t2.shape == (22,)

    def test_silence(self, silence_3s):
        feats = extract_all_features(silence_3s, SR)
        for key in ("tier1", "tier2", "combined"):
            assert not np.any(np.isnan(feats[key])), f"NaN in {key} for silence"
            assert not np.any(np.isinf(feats[key])), f"Inf in {key} for silence"


# -- Chunking Tests --


class TestChunking:
    def test_chunk_count(self):
        audio = np.random.default_rng(0).standard_normal(30 * SR).astype(np.float32)
        chunks = chunk_audio(audio, SR, window_sec=10.0, stride_sec=5.0)
        assert len(chunks) == 5

    def test_chunk_short_audio(self):
        audio = np.random.default_rng(0).standard_normal(3 * SR).astype(np.float32)
        chunks = chunk_audio(audio, SR, window_sec=10.0, stride_sec=5.0)
        assert len(chunks) == 1

    def test_chunk_sizes(self):
        audio = np.random.default_rng(0).standard_normal(30 * SR).astype(np.float32)
        window_sec = 10.0
        chunks = chunk_audio(audio, SR, window_sec=window_sec, stride_sec=5.0)
        expected_samples = int(window_sec * SR)
        for chunk in chunks:
            assert len(chunk) == expected_samples

    def test_chunk_padding(self):
        audio = np.ones(12 * SR, dtype=np.float32)
        chunks = chunk_audio(audio, SR, window_sec=10.0, stride_sec=5.0)
        last = chunks[-1]
        window_samples = int(10.0 * SR)
        assert len(last) == window_samples
        real_samples = 12 * SR - 5 * SR
        assert np.all(last[:real_samples] == 1.0)
        assert np.all(last[real_samples:] == 0.0)


# -- Model Tests --


class TestModels:
    def test_attention_pooling_shape(self):
        pool = AttentionPooling(embed_dim=256)
        x = torch.randn(2, 5, 256)
        out = pool(x)
        assert out.shape == (2, 256)

    def test_feature_projector_shape(self):
        proj = FeatureProjector(feature_dim=452, embed_dim=256)
        x = torch.randn(2, 452)
        out = proj(x)
        assert out.shape == (2, 256)

    def test_siamese_output_shape(self):
        net = SiameseNetwork(feature_dim=452, embed_dim=256)
        x1 = torch.randn(2, 5, 452)
        x2 = torch.randn(2, 5, 452)
        emb1, emb2 = net(x1, x2)
        assert emb1.shape == (2, 256)
        assert emb2.shape == (2, 256)

    def test_similarity_head_output_range(self):
        torch.manual_seed(42)
        head = SimilarityHead(embed_dim=256, hidden_dim=128)
        head.eval()
        emb_a = torch.randn(4, 256)
        emb_b = torch.randn(4, 256)
        with torch.no_grad():
            logits = head(emb_a, emb_b)
        assert logits.shape == (4,)
        # Head now returns raw logits; sigmoid is applied in PairwiseSimilarityModel
        probs = torch.sigmoid(logits)
        assert torch.all(probs >= 0.0).item()
        assert torch.all(probs <= 1.0).item()

    def test_pairwise_model_end_to_end(self):
        torch.manual_seed(42)
        model = PairwiseSimilarityModel(feature_dim=452, embed_dim=256, hidden_dim=128)
        model.eval()
        x1 = torch.randn(2, 5, 452)
        x2 = torch.randn(2, 5, 452)
        with torch.no_grad():
            scores = model(x1, x2)
        assert scores.shape == (2,)

    def test_identical_inputs_high_similarity(self):
        torch.manual_seed(42)
        model = PairwiseSimilarityModel(feature_dim=452, embed_dim=256, hidden_dim=128)
        model.eval()
        x = torch.randn(2, 5, 452)
        with torch.no_grad():
            scores = model(x, x)
        assert torch.all(scores > 0.4).item()

    def test_multi_head_attention_pooling_shape(self):
        pool = MultiHeadAttentionPooling(embed_dim=256, n_heads=4)
        x = torch.randn(2, 5, 256)
        out = pool(x)
        assert out.shape == (2, 256)

    def test_projection_head_shape(self):
        proj = ProjectionHead(embed_dim=256, proj_dim=128)
        x = torch.randn(4, 256)
        out = proj(x)
        assert out.shape == (4, 128)

    def test_siamese_projection(self):
        net = SiameseNetwork(feature_dim=452, embed_dim=256, use_projection=True, proj_dim=128)
        x1 = torch.randn(2, 5, 452)
        x2 = torch.randn(2, 5, 452)
        emb1, emb2 = net(x1, x2)
        proj1 = net.project(emb1)
        proj2 = net.project(emb2)
        assert proj1.shape == (2, 128)
        assert proj2.shape == (2, 128)

    def test_focal_loss_runs(self):
        focal = FocalLoss(gamma=2.0, alpha=0.25)
        logits = torch.randn(8)
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        loss = focal(logits, labels)
        assert loss.dim() == 0  # scalar
        assert loss.item() > 0

    def test_focal_loss_with_label_smoothing(self):
        focal = FocalLoss(gamma=2.0, alpha=0.25, label_smoothing=0.1)
        logits = torch.randn(8)
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        loss = focal(logits, labels)
        assert loss.item() > 0

    def test_deeper_similarity_head(self):
        """Verify 3-layer head with residual produces correct shape."""
        torch.manual_seed(42)
        head = SimilarityHead(embed_dim=256, hidden_dim=128)
        emb_a = torch.randn(4, 256)
        emb_b = torch.randn(4, 256)
        out = head(emb_a, emb_b)
        assert out.shape == (4,)

    def test_projection_head_uses_layernorm(self):
        """ProjectionHead should use LayerNorm (not BatchNorm) for train/eval consistency."""
        proj = ProjectionHead(embed_dim=256, proj_dim=128)
        norm_layers = [m for m in proj.modules() if isinstance(m, torch.nn.LayerNorm)]
        bn_layers = [m for m in proj.modules() if isinstance(m, torch.nn.BatchNorm1d)]
        assert len(norm_layers) == 1
        assert len(bn_layers) == 0

    def test_l2_normalized_projections(self):
        """Projected + L2-normalized embeddings should have unit norm."""
        net = SiameseNetwork(feature_dim=452, embed_dim=256, use_projection=True, proj_dim=128)
        x = torch.randn(4, 5, 452)
        emb, _ = net(x, x)
        proj = net.project(emb)
        proj_norm = torch.nn.functional.normalize(proj, dim=-1)
        norms = proj_norm.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestFeatureStats:
    def test_compute_feature_stats(self, tmp_path):
        """compute_feature_stats should produce correct mean/std from .npy files."""
        rng = np.random.default_rng(42)
        # Create synthetic feature files with known statistics
        for i in range(10):
            arr = rng.standard_normal((5, 20)).astype(np.float32) * 3.0 + 7.0
            np.save(tmp_path / f"{i:04d}.npy", arr)

        stats = compute_feature_stats(tmp_path)
        assert "mean" in stats and "std" in stats
        assert stats["mean"].shape == (20,)
        assert stats["std"].shape == (20,)
        # Mean should be ~7.0, std ~3.0 (50 vectors total)
        assert np.allclose(stats["mean"], 7.0, atol=1.0)
        assert np.allclose(stats["std"], 3.0, atol=1.0)
        # Check .npz file was saved
        assert (tmp_path / "feature_stats.npz").exists()

    def test_compute_feature_stats_empty_dir(self, tmp_path):
        """Should raise FileNotFoundError with no .npy files."""
        with pytest.raises(FileNotFoundError):
            compute_feature_stats(tmp_path)

    def test_pad_or_truncate_training_random_offset(self, tmp_path):
        """In training mode _pad_or_truncate should use a random start offset."""
        from src.models.pair_dataset import AudioPairDataset

        # Build a minimal pairs CSV so we can instantiate the dataset
        import pandas as pd
        csv_path = tmp_path / "pairs.csv"
        pd.DataFrame({"track_a_path": ["a.wav"], "track_b_path": ["b.wav"], "label": [1]}).to_csv(csv_path, index=False)

        ds_train = AudioPairDataset(str(csv_path), training=True)
        ds_eval = AudioPairDataset(str(csv_path), training=False)

        # Create a features array with num_chunks > max_chunks (30 > 12)
        rng = np.random.default_rng(99)
        feats = rng.standard_normal((30, 16)).astype(np.float32)

        # Eval mode always returns the first 12 rows
        eval_result = ds_eval._pad_or_truncate(feats)
        assert eval_result.shape == (12, 16)
        np.testing.assert_array_equal(eval_result, feats[:12])

        # Training mode should occasionally return a different start offset
        starts = set()
        for _ in range(50):
            result = ds_train._pad_or_truncate(feats)
            assert result.shape == (12, 16)
            # find the start by matching the first row
            for s in range(30 - 12 + 1):
                if np.allclose(feats[s], result[0]):
                    starts.add(s)
                    break
        # With 50 tries and max_start=18, almost certain to see more than 1 offset
        assert len(starts) > 1, "Training mode should sample different start offsets"


# -- Similarity Function Tests --


class TestSimilarityFunctions:
    def test_melodic_similarity_range(self, sine_3s, white_noise_3s):
        score = melodic_similarity(sine_3s, white_noise_3s, SR)
        assert 0.0 <= score <= 1.0

    def test_timbral_similarity_range(self, sine_3s, white_noise_3s):
        score = timbral_similarity(sine_3s, white_noise_3s, SR)
        assert 0.0 <= score <= 1.0

    def test_structural_similarity_range(self, sine_3s, white_noise_3s):
        score = structural_similarity(sine_3s, white_noise_3s, SR)
        assert 0.0 <= score <= 1.0

    def test_embedding_similarity_identical(self):
        emb = np.random.default_rng(0).standard_normal(256).astype(np.float32)
        score = embedding_similarity(emb, emb)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_ai_artifact_score_range(self, sine_3s):
        score = compute_ai_artifact_score(sine_3s, SR)
        assert 0.0 <= score <= 1.0

    def test_self_similarity_high(self, sine_3s):
        mel = melodic_similarity(sine_3s, sine_3s, SR)
        tim = timbral_similarity(sine_3s, sine_3s, SR)
        struc = structural_similarity(sine_3s, sine_3s, SR)
        assert mel > 0.8
        assert tim > 0.8
        assert struc > 0.8
