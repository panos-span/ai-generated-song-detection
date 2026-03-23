from __future__ import annotations

import torch
import torch.nn as nn

from src.models.siamese_network import SegmentAwareSiamese, SiameseNetwork


class SimilarityHead(nn.Module):
    """MLP head for computing similarity from paired embeddings.

    Combines embeddings as: [emb_a; emb_b; |emb_a - emb_b|; emb_a * emb_b]
    Then passes through 3-layer MLP with residual -> logits (raw, no sigmoid).
    """

    def __init__(self, embed_dim: int = 256, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        input_dim = embed_dim * 4
        mid_dim = hidden_dim // 2
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Residual projection: hidden_dim -> mid_dim
        self.residual_proj = nn.Linear(hidden_dim, mid_dim)
        self.output = nn.Linear(mid_dim, 1)

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        """Returns similarity logits shape (batch,)"""
        combined = torch.cat(
            [emb_a, emb_b, torch.abs(emb_a - emb_b), emb_a * emb_b],
            dim=-1,
        )
        h = self.layer1(combined)
        h2 = self.layer2(h) + self.residual_proj(h)
        return self.output(h2).squeeze(-1)


class PairwiseSimilarityModel(nn.Module):
    """Complete model combining SiameseNetwork + SimilarityHead.

    Supports optional architecture variants:
    - ``use_spectrogram``: dual-stream with spectrogram CNN (Phase 3)
    - ``use_segment_transformer``: SegmentAwareSiamese replacing AttentionPooling (Phase 4)
    """

    def __init__(
        self,
        feature_dim: int,
        embed_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        use_spectrogram: bool = False,
        use_segment_transformer: bool = False,
    ) -> None:
        super().__init__()
        if use_segment_transformer:
            self.siamese = SegmentAwareSiamese(feature_dim, embed_dim, dropout)
        else:
            self.siamese = SiameseNetwork(
                feature_dim, embed_dim, dropout, use_spectrogram=use_spectrogram,
            )
        self.head = SimilarityHead(embed_dim, hidden_dim, dropout)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """x1, x2: (batch, num_chunks, feature_dim) -> scores (batch,) in [0, 1]"""
        emb1, emb2 = self.siamese(x1, x2)
        return torch.sigmoid(self.head(emb1, emb2))


class CoarseToFineHead(nn.Module):
    """Two-stage comparison inspired by CoverHunter (C1).

    Coarse: Global embedding cosine similarity (fast rejection if < threshold).
    Fine: Learned attention alignment on per-chunk embedding sequences.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dim: int = 128,
        coarse_threshold: float = 0.3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.coarse_threshold = coarse_threshold
        self.fine_head = SimilarityHead(embed_dim, hidden_dim, dropout)
        self.cross_attn = nn.MultiheadAttention(embed_dim, 4, dropout=dropout, batch_first=True)
        self.pool = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )

    def coarse_score(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        """Fast cosine similarity between global embeddings. (batch,)"""
        a_norm = torch.nn.functional.normalize(emb_a, dim=-1)
        b_norm = torch.nn.functional.normalize(emb_b, dim=-1)
        return (a_norm * b_norm).sum(dim=-1)

    def fine_score(
        self, seq_a: torch.Tensor, seq_b: torch.Tensor,
    ) -> torch.Tensor:
        """Learned alignment score from chunk sequences. (batch,)

        seq_a, seq_b: (batch, num_chunks, embed_dim)
        """
        aligned, _ = self.cross_attn(seq_a, seq_b, seq_b)
        diff = torch.abs(seq_a - aligned)
        scores = self.pool(diff).squeeze(-1)
        return torch.sigmoid(scores.mean(dim=-1))

    def forward(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        seq_a: torch.Tensor | None = None,
        seq_b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Combined coarse + fine scoring.

        If seq_a/seq_b not provided, falls back to standard SimilarityHead.
        """
        if seq_a is None or seq_b is None:
            return torch.sigmoid(self.fine_head(emb_a, emb_b))

        coarse = self.coarse_score(emb_a, emb_b)

        # In training, always compute fine score for gradient flow
        if self.training:
            fine = self.fine_score(seq_a, seq_b)
            return 0.3 * coarse + 0.7 * fine

        # In inference, skip fine stage for clearly dissimilar pairs
        fine = self.fine_score(seq_a, seq_b)
        return 0.3 * coarse + 0.7 * fine
