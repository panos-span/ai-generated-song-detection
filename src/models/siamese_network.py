from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.models.spectrogram_encoder import SpectrogramEncoder


class AttentionPooling(nn.Module):
    """Attention-based aggregation of chunk-level embeddings into a single track embedding."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(embed_dim))
        self.scale = embed_dim ** 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, num_chunks, embed_dim) -> (batch, embed_dim)"""
        scores = torch.matmul(x, self.query) / self.scale  # (batch, num_chunks)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (batch, num_chunks, 1)
        return (x * weights).sum(dim=1)  # (batch, embed_dim)


class MultiHeadAttentionPooling(nn.Module):
    """Multi-head attention pooling that captures diverse aspects of the sequence.

    Each head learns a different query vector, attending to different aspects
    (e.g. timbre, rhythm, structure). Outputs are concatenated and projected
    back to embed_dim.
    """

    def __init__(self, embed_dim: int, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.queries = nn.Parameter(torch.randn(n_heads, embed_dim))
        self.scale = embed_dim ** 0.5
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * n_heads, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, num_chunks, embed_dim) -> (batch, embed_dim)"""
        # scores: (n_heads, batch, num_chunks)
        scores = torch.einsum("bse,he->hbs", x, self.queries) / self.scale
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (n_heads, batch, num_chunks, 1)
        # pooled: (n_heads, batch, embed_dim)
        pooled = (x.unsqueeze(0) * weights).sum(dim=2)
        # concat heads: (batch, n_heads * embed_dim)
        multi = pooled.permute(1, 0, 2).reshape(x.size(0), -1)
        return self.proj(multi)


class ProjectionHead(nn.Module):
    """MLP projection for contrastive/triplet losses (SimCLR-style).

    Maps embeddings to a lower-dimensional space for metric loss computation.
    Discarded at inference time (zero overhead).
    """

    def __init__(self, embed_dim: int = 256, proj_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeatureProjector(nn.Module):
    """Projects raw feature vectors to a fixed embedding dimension."""

    def __init__(self, feature_dim: int, embed_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SiameseNetwork(nn.Module):
    """Siamese network for pairwise audio similarity."""

    def __init__(
        self,
        feature_dim: int,
        embed_dim: int = 256,
        dropout: float = 0.1,
        use_spectrogram: bool = False,
        use_projection: bool = True,
        proj_dim: int = 128,
    ) -> None:
        super().__init__()
        self.projector = FeatureProjector(feature_dim, embed_dim, dropout)
        self.use_spectrogram = use_spectrogram

        if use_spectrogram:
            self.spec_encoder = SpectrogramEncoder(embed_dim=embed_dim)
            self.fusion = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
            )

        self.attention = MultiHeadAttentionPooling(embed_dim, n_heads=4, dropout=dropout)

        # Projection head for contrastive/triplet losses (training only)
        self.projection: ProjectionHead | None = None
        if use_projection:
            self.projection = ProjectionHead(embed_dim, proj_dim)

    def forward_one(
        self, x: torch.Tensor, spec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Process one track.

        Args:
            x: (batch, num_chunks, feature_dim)
            spec: optional (batch, num_chunks, 1, n_mels, time_frames)
        """
        batch, num_chunks, _ = x.shape
        flat = x.reshape(batch * num_chunks, -1)
        projected = self.projector(flat)

        if self.use_spectrogram and spec is not None:
            spec_flat = spec.reshape(batch * num_chunks, *spec.shape[2:])
            spec_emb = self.spec_encoder(spec_flat)
            projected = self.fusion(torch.cat([projected, spec_emb], dim=-1))

        projected = projected.reshape(batch, num_chunks, -1)
        return self.attention(projected)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        spec1: torch.Tensor | None = None,
        spec2: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (emb1, emb2) both shape (batch, embed_dim)"""
        return self.forward_one(x1, spec1), self.forward_one(x2, spec2)

    def project(self, emb: torch.Tensor) -> torch.Tensor:
        """Project embedding for contrastive/triplet loss. Falls back to identity."""
        if self.projection is not None:
            return self.projection(emb)
        return emb


# ---------------------------------------------------------------------------
# Phase 3: Dual-Stream Encoder (CLAM A4, Deezer TwoStreamLitMLP)
# ---------------------------------------------------------------------------


class CrossAggregation(nn.Module):
    """Multi-head cross-attention between two stream outputs (CLAM paper A4)."""

    def __init__(self, embed_dim: int = 256, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query_stream: torch.Tensor, kv_stream: torch.Tensor) -> torch.Tensor:
        """query_stream, kv_stream: (batch, seq_len, embed_dim) -> (batch, seq_len, embed_dim)"""
        attn_out, _ = self.cross_attn(query_stream, kv_stream, kv_stream)
        return self.norm(query_stream + attn_out)


class GatedFusion(nn.Module):
    """Learnable gated fusion of two embedding streams (A5 Fusion Segment Transformer).

    output = gate * stream1 + (1 - gate) * stream2
    where gate = sigmoid(Linear(concat(stream1, stream2)))
    """

    def __init__(self, embed_dim: int = 256) -> None:
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )

    def forward(self, stream1: torch.Tensor, stream2: torch.Tensor) -> torch.Tensor:
        gate = self.gate_net(torch.cat([stream1, stream2], dim=-1))
        return gate * stream1 + (1 - gate) * stream2


class DualStreamEncoder(nn.Module):
    """Dual-stream encoder combining two embedding sources.

    Stream 1: Primary embeddings (e.g. MERT 768-d)
    Stream 2: Secondary embeddings (e.g. Wav2Vec2 1024-d or Whisper 1280-d)

    Fusion modes:
      - "concat": Concatenation + MLP (Deezer TwoStreamLitMLP pattern)
      - "cross_attention": Cross-aggregation (CLAM A4)
      - "gated": Gated fusion (A5)
    """

    def __init__(
        self,
        stream1_dim: int = 768,
        stream2_dim: int = 1024,
        embed_dim: int = 256,
        fusion_mode: str = "gated",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.proj1 = nn.Sequential(
            nn.Linear(stream1_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.proj2 = nn.Sequential(
            nn.Linear(stream2_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion_mode = fusion_mode

        if fusion_mode == "concat":
            self.fuse = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
            )
        elif fusion_mode == "cross_attention":
            self.cross_agg = CrossAggregation(embed_dim, dropout=dropout)
        elif fusion_mode == "gated":
            self.gated = GatedFusion(embed_dim)

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        """emb1: (batch, stream1_dim), emb2: (batch, stream2_dim) -> (batch, embed_dim)"""
        p1 = self.proj1(emb1)
        p2 = self.proj2(emb2)

        if self.fusion_mode == "concat":
            return self.fuse(torch.cat([p1, p2], dim=-1))
        elif self.fusion_mode == "cross_attention":
            p1_seq = p1.unsqueeze(1)
            p2_seq = p2.unsqueeze(1)
            fused = self.cross_agg(p1_seq, p2_seq)
            return fused.squeeze(1)
        elif self.fusion_mode == "gated":
            return self.gated(p1, p2)
        else:
            return p1 + p2


# ---------------------------------------------------------------------------
# Phase 4: Segment-Level Structural Analysis (A5, A6, C1)
# ---------------------------------------------------------------------------


class SegmentTransformer(nn.Module):
    """Small Transformer encoder for inter-chunk dependency modeling.

    Takes per-chunk embeddings as a token sequence and models temporal
    relationships between segments, replacing simple AttentionPooling.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
        max_chunks: int = 64,
    ) -> None:
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_chunks, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pool_query = nn.Parameter(torch.randn(embed_dim))
        self.scale = math.sqrt(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, num_chunks, embed_dim) -> (batch, embed_dim)"""
        seq_len = x.shape[1]
        x = x + self.pos_embedding[:, :seq_len, :]
        x = self.transformer(x)
        # Attention-weighted pooling over transformer outputs
        scores = torch.matmul(x, self.pool_query) / self.scale
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (x * weights).sum(dim=1)


class SegmentAwareSiamese(nn.Module):
    """Siamese network with SegmentTransformer + GatedFusion (Phase 4).

    Replaces AttentionPooling with segment-aware transformer and optional
    structural embedding fusion. Backward compatible with --use-segment-transformer flag.
    """

    def __init__(
        self,
        feature_dim: int,
        embed_dim: int = 256,
        dropout: float = 0.1,
        n_transformer_layers: int = 2,
        use_structural_fusion: bool = True,
    ) -> None:
        super().__init__()
        self.projector = FeatureProjector(feature_dim, embed_dim, dropout)
        self.segment_transformer = SegmentTransformer(
            embed_dim=embed_dim,
            n_layers=n_transformer_layers,
            dropout=dropout,
        )
        self.use_structural_fusion = use_structural_fusion
        if use_structural_fusion:
            self.structural_proj = nn.Linear(3, embed_dim)
            self.gated_fusion = GatedFusion(embed_dim)

    def forward_one(self, x: torch.Tensor, structural: torch.Tensor | None = None) -> torch.Tensor:
        """Process one track.

        Args:
            x: (batch, num_chunks, feature_dim)
            structural: optional (batch, 3) -- SSM novelty, repetition, transition
        """
        batch, num_chunks, _ = x.shape
        flat = x.reshape(batch * num_chunks, -1)
        projected = self.projector(flat).reshape(batch, num_chunks, -1)
        content_emb = self.segment_transformer(projected)

        if self.use_structural_fusion and structural is not None:
            struct_emb = self.structural_proj(structural)
            return self.gated_fusion(content_emb, struct_emb)
        return content_emb

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        struct1: torch.Tensor | None = None,
        struct2: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_one(x1, struct1), self.forward_one(x2, struct2)
