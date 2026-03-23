"""Phase 6 – SimCLR / NT-Xent contrastive pre-training for audio embeddings.

Pre-trains a SiameseNetwork encoder using augmented views of the same track
as positive pairs (paper C4).  The learned representations can be fine-tuned
downstream for pairwise similarity scoring.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.features.augmentations import AudioAugmentor
from src.log_config import setup_logging
from src.models.siamese_network import SiameseNetwork

torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# NT-Xent loss (normalized temperature-scaled cross-entropy)
# ---------------------------------------------------------------------------


class NTXentLoss(nn.Module):
    """Normalized temperature-scaled cross-entropy loss (SimCLR).

    Given two batches of L2-normalised embeddings ``z_i`` and ``z_j`` that
    form positive pairs row-wise, the loss maximises agreement between each
    positive pair while treating every other sample in the batch as a
    negative.

    Parameters
    ----------
    temperature:
        Scaling factor for the cosine similarity logits.  Lower values
        sharpen the distribution and increase gradient magnitude.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """Compute the NT-Xent loss.

        Args:
            z_i: Embeddings of view 1, shape ``(batch, embed_dim)``.
            z_j: Embeddings of view 2, shape ``(batch, embed_dim)``.

        Returns:
            Scalar loss tensor.
        """
        batch_size = z_i.shape[0]

        # L2-normalise so dot product == cosine similarity
        z_i = nn.functional.normalize(z_i, dim=-1)
        z_j = nn.functional.normalize(z_j, dim=-1)

        # Full similarity matrix of shape (2N, 2N)
        representations = torch.cat([z_i, z_j], dim=0)  # (2N, D)
        sim_matrix = torch.mm(representations, representations.t()) / self.temperature

        # Mask out self-similarity (diagonal)
        mask = torch.eye(2 * batch_size, device=sim_matrix.device, dtype=torch.bool)
        sim_matrix.masked_fill_(mask, float("-inf"))

        # Positive-pair indices: (i, i+N) and (i+N, i) for each i in [0, N)
        pos_indices = torch.arange(batch_size, device=sim_matrix.device)
        labels = torch.cat([pos_indices + batch_size, pos_indices], dim=0)

        loss = nn.functional.cross_entropy(sim_matrix, labels)
        return loss


# ---------------------------------------------------------------------------
# Contrastive pre-trainer
# ---------------------------------------------------------------------------


class ContrastivePretrainer:
    """Pre-trains a model encoder with NT-Xent contrastive learning.

    Positive pairs are created on-the-fly by applying two independent
    stochastic augmentation views of the same input track.  The encoder is
    trained to produce similar embeddings for the two views.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        lr: float = 1e-4,
        device: str = "auto",
        temperature: float = 0.1,
        weight_decay: float = 1e-2,
    ) -> None:
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)

        try:
            self.model = torch.compile(self.model)
        except Exception:
            logger.warning("torch.compile() not supported in this environment; skipping")

        self.train_loader = train_loader
        self.criterion = NTXentLoss(temperature=temperature)
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    # ------------------------------------------------------------------
    # Training steps
    # ------------------------------------------------------------------

    def pretrain_epoch(self) -> dict[str, float]:
        """Run one contrastive pre-training epoch.

        The dataloader is expected to yield dicts with keys ``view_i`` and
        ``view_j``, each of shape ``(batch, num_chunks, feature_dim)``.

        Returns:
            Dict with ``loss`` (average NT-Xent loss over the epoch).
        """
        self.model.train()
        running_loss = 0.0
        total = 0

        self.optimizer.zero_grad(set_to_none=True)

        for batch in tqdm(self.train_loader, desc="Pre-training", leave=False):
            view_i = batch["view_i"].to(self.device)
            view_j = batch["view_j"].to(self.device)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                # Encode both views through the shared encoder
                emb_i = self.model.forward_one(view_i)
                emb_j = self.model.forward_one(view_j)
                loss = self.criterion(emb_i, emb_j)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            bs = view_i.size(0)
            running_loss += loss.item() * bs
            total += bs

        n = max(total, 1)
        return {"loss": running_loss / n}

    # ------------------------------------------------------------------
    # Full pre-training loop
    # ------------------------------------------------------------------

    def pretrain(
        self,
        num_epochs: int = 100,
        save_dir: str = "pretrained",
    ) -> dict[str, float]:
        """Run the full contrastive pre-training loop and save the best checkpoint.

        Parameters
        ----------
        num_epochs:
            Maximum number of pre-training epochs.
        save_dir:
            Directory for saving the best encoder checkpoint.

        Returns:
            Dict containing the best epoch's metrics.
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        best_loss = float("inf")
        best_epoch = 0
        best_metrics: dict[str, float] = {}

        for epoch in range(1, num_epochs + 1):
            metrics = self.pretrain_epoch()
            is_best = metrics["loss"] < best_loss

            if is_best:
                best_loss = metrics["loss"]
                best_epoch = epoch
                best_metrics = {**metrics, "epoch": float(epoch)}
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "loss": best_loss,
                    },
                    save_path / "pretrained_best.pt",
                )

            marker = " *" if is_best else ""
            logger.info(
                "Epoch %3d/%d | nt_xent_loss=%.4f | best=%.4f (ep%d)%s",
                epoch,
                num_epochs,
                metrics["loss"],
                best_loss,
                best_epoch,
                marker,
            )

        logger.info(
            "Pre-training complete. Best loss %.4f at epoch %d",
            best_loss,
            best_epoch,
        )
        return best_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 6: SimCLR contrastive pre-training for audio embeddings",
    )
    parser.add_argument("--data_dir", required=True, help="Root directory containing audio files")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of pre-training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--temperature", type=float, default=0.1, help="NT-Xent temperature")
    parser.add_argument(
        "--output_dir",
        default="pretrained",
        help="Directory for saving pretrained checkpoint",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--feature_dim", type=int, default=452, help="Input feature dimension")
    parser.add_argument("--aug_prob", type=float, default=0.5, help="Per-augmentation probability")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="DataLoader workers (default: min(8, cpu_count))",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="AdamW weight decay")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    setup_logging()

    num_workers = args.num_workers if args.num_workers is not None else min(8, os.cpu_count() or 4)
    augmentor = AudioAugmentor(sr=16000, p=args.aug_prob, enabled=True)
    logger.info("AudioAugmentor enabled (p={})", args.aug_prob)

    # ------------------------------------------------------------------
    # Build a contrastive dataset that yields two augmented views per track.
    # We reuse AudioPairDataset in self-pair mode: each track is paired with
    # itself and augmented independently via the augmentor.
    # ------------------------------------------------------------------
    from src.models.pair_dataset import AudioPairDataset, collate_pairs

    pairs_csv = Path(args.data_dir) / "train_pairs.csv"
    if not pairs_csv.exists():
        logger.error("Expected training pairs CSV at %s", pairs_csv)
        raise FileNotFoundError(pairs_csv)

    dataset = AudioPairDataset(
        str(pairs_csv),
        augmentor=augmentor,
    )
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=collate_pairs,
    )

    # Wrap the loader to yield view_i / view_j keys expected by the pretrainer.
    # Both views come from the same underlying pair (with stochastic augmentation
    # applied independently each time the dataset is indexed).
    class _ViewAdapter:
        """Adapts AudioPairDataset batches to view_i / view_j dicts."""

        def __init__(self, loader: DataLoader) -> None:
            self.loader = loader

        def __iter__(self):  # noqa: ANN204
            for batch in self.loader:
                yield {
                    "view_i": batch["features_a"],
                    "view_j": batch["features_b"],
                }

        def __len__(self) -> int:
            return len(self.loader)

    adapted_loader = _ViewAdapter(train_loader)

    model = SiameseNetwork(
        feature_dim=args.feature_dim,
        embed_dim=args.embed_dim,
    )

    pretrainer = ContrastivePretrainer(
        model=model,
        train_loader=adapted_loader,
        lr=args.lr,
        device=args.device,
        temperature=args.temperature,
        weight_decay=args.weight_decay,
    )

    best_metrics = pretrainer.pretrain(
        num_epochs=args.epochs,
        save_dir=args.output_dir,
    )
    logger.info("Pre-training finished. Best metrics: %s", best_metrics)


if __name__ == "__main__":
    main()