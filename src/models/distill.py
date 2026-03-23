"""Phase 7 – Knowledge distillation from teacher encoder to lightweight student.

Implements the Re-MOVE (C2) distillation strategy: a large pre-trained teacher
encoder produces reference embeddings, and a small CNN student is trained to
reproduce them via MSE loss.  The resulting student model is fast enough for
real-time or large-scale retrieval while retaining most of the teacher's
discriminative power.
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

from src.log_config import setup_logging
from src.models.siamese_network import SiameseNetwork

torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# Lightweight student encoder
# ---------------------------------------------------------------------------


class StudentEncoder(nn.Module):
    """Small CNN student for embedding distillation.

    Architecture: 3 × Conv1d layers with BatchNorm + ReLU, followed by
    adaptive average pooling and a linear projection to the target embedding
    dimension.

    Parameters
    ----------
    feature_dim:
        Dimensionality of each input chunk feature vector.
    embed_dim:
        Output embedding dimension (should match the teacher for MSE loss).
    """

    def __init__(self, feature_dim: int, embed_dim: int = 128) -> None:
        super().__init__()
        self.conv_layers = nn.Sequential(
            # Conv block 1
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            # Conv block 2
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            # Conv block 3
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(256, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of feature vectors.

        Args:
            x: Input features of shape ``(batch, feature_dim)``.

        Returns:
            Embeddings of shape ``(batch, embed_dim)``.
        """
        # Treat feature_dim as the temporal axis for Conv1d: (batch, 1, feature_dim)
        x = x.unsqueeze(1)
        x = self.conv_layers(x)        # (batch, 256, feature_dim)
        x = self.pool(x).squeeze(-1)   # (batch, 256)
        x = self.projection(x)         # (batch, embed_dim)
        return x


# ---------------------------------------------------------------------------
# Distillation trainer
# ---------------------------------------------------------------------------


class DistillationTrainer:
    """Trains a lightweight student to mimic teacher embeddings via MSE loss.

    The teacher model is frozen throughout training; only the student's
    parameters are updated.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        train_loader: DataLoader,
        lr: float = 1e-3,
        device: str = "auto",
        weight_decay: float = 1e-2,
    ) -> None:
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Teacher is frozen — eval mode, no gradients
        self.teacher = teacher.to(self.device)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.student = student.to(self.device)

        try:
            self.student = torch.compile(self.student)
        except Exception:
            logger.warning("torch.compile() not supported in this environment; skipping")

        self.train_loader = train_loader
        self.mse_fn = nn.MSELoss()
        self.optimizer = AdamW(self.student.parameters(), lr=lr, weight_decay=weight_decay)

        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    # ------------------------------------------------------------------
    # Training steps
    # ------------------------------------------------------------------

    def train_epoch(self) -> dict[str, float]:
        """Run one distillation epoch.

        The dataloader is expected to yield dicts with key ``features``
        of shape ``(batch, num_chunks, feature_dim)``.

        Returns:
            Dict with ``mse_loss`` (average MSE over the epoch).
        """
        self.student.train()
        running_loss = 0.0
        total = 0

        self.optimizer.zero_grad(set_to_none=True)

        for batch in tqdm(self.train_loader, desc="Distillation", leave=False):
            features = batch["features"].to(self.device)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                # Teacher embedding (no grad)
                with torch.no_grad():
                    teacher_emb = self.teacher.forward_one(features)

                # Student operates on the mean-pooled chunk features
                # (batch, num_chunks, feature_dim) -> (batch, feature_dim)
                flat_features = features.mean(dim=1)
                student_emb = self.student(flat_features)

                loss = self.mse_fn(student_emb, teacher_emb)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            bs = features.size(0)
            running_loss += loss.item() * bs
            total += bs

        n = max(total, 1)
        return {"mse_loss": running_loss / n}

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(
        self,
        num_epochs: int = 50,
        save_dir: str = "distilled",
    ) -> dict[str, float]:
        """Run full distillation training and save the best student checkpoint.

        Parameters
        ----------
        num_epochs:
            Maximum number of training epochs.
        save_dir:
            Directory for saving the best student checkpoint.

        Returns:
            Dict containing the best epoch's metrics.
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        best_loss = float("inf")
        best_epoch = 0
        best_metrics: dict[str, float] = {}

        for epoch in range(1, num_epochs + 1):
            metrics = self.train_epoch()
            is_best = metrics["mse_loss"] < best_loss

            if is_best:
                best_loss = metrics["mse_loss"]
                best_epoch = epoch
                best_metrics = {**metrics, "epoch": float(epoch)}
                torch.save(
                    {
                        "epoch": epoch,
                        "student_state_dict": self.student.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "mse_loss": best_loss,
                    },
                    save_path / "student_best.pt",
                )

            marker = " *" if is_best else ""
            logger.info(
                "Epoch %3d/%d | mse_loss=%.6f | best=%.6f (ep%d)%s",
                epoch,
                num_epochs,
                metrics["mse_loss"],
                best_loss,
                best_epoch,
                marker,
            )

        logger.info(
            "Distillation complete. Best MSE %.6f at epoch %d",
            best_loss,
            best_epoch,
        )
        return best_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 7: Knowledge distillation from teacher to lightweight student",
    )
    parser.add_argument(
        "--teacher_path",
        required=True,
        help="Path to trained teacher checkpoint (.pt)",
    )
    parser.add_argument("--data_dir", required=True, help="Root directory containing audio data")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--student_dim",
        type=int,
        default=128,
        help="Student output embedding dimension",
    )
    parser.add_argument(
        "--output_dir",
        default="distilled",
        help="Directory for saving distilled student checkpoint",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--feature_dim", type=int, default=452, help="Input feature dimension")
    parser.add_argument("--embed_dim", type=int, default=256, help="Teacher embedding dimension")
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

    # ------------------------------------------------------------------
    # Load teacher model from checkpoint
    # ------------------------------------------------------------------
    teacher_ckpt_path = Path(args.teacher_path)
    if not teacher_ckpt_path.exists():
        logger.error("Teacher checkpoint not found: %s", teacher_ckpt_path)
        raise FileNotFoundError(teacher_ckpt_path)

    checkpoint = torch.load(str(teacher_ckpt_path), map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Infer feature_dim from checkpoint if possible
    proj_key = "projector.net.0.weight"
    if proj_key in state_dict:
        feature_dim = state_dict[proj_key].shape[1]
        logger.info("Inferred feature_dim=%d from teacher checkpoint", feature_dim)
    else:
        feature_dim = args.feature_dim
        logger.info("Using CLI feature_dim=%d", feature_dim)

    teacher = SiameseNetwork(feature_dim=feature_dim, embed_dim=args.embed_dim)
    teacher.load_state_dict(state_dict, strict=False)
    logger.info("Teacher model loaded from %s", teacher_ckpt_path)

    # ------------------------------------------------------------------
    # Build student model
    # ------------------------------------------------------------------
    student = StudentEncoder(feature_dim=feature_dim, embed_dim=args.student_dim)
    student_params = sum(p.numel() for p in student.parameters())
    teacher_params = sum(p.numel() for p in teacher.parameters())
    logger.info(
        "Student: %d params | Teacher: %d params | Compression: %.1fx",
        student_params,
        teacher_params,
        teacher_params / max(student_params, 1),
    )

    # ------------------------------------------------------------------
    # Build dataloader
    # ------------------------------------------------------------------
    from src.models.pair_dataset import AudioPairDataset, collate_pairs

    pairs_csv = Path(args.data_dir) / "train_pairs.csv"
    if not pairs_csv.exists():
        logger.error("Expected training pairs CSV at %s", pairs_csv)
        raise FileNotFoundError(pairs_csv)

    dataset = AudioPairDataset(str(pairs_csv))
    raw_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=collate_pairs,
    )

    # Adapter: distillation only needs single-track features, so we use
    # features_a from each pair as the training signal.
    class _SingleTrackAdapter:
        """Adapts AudioPairDataset batches to single-track feature dicts."""

        def __init__(self, loader: DataLoader) -> None:
            self.loader = loader

        def __iter__(self):  # noqa: ANN204
            for batch in self.loader:
                yield {"features": batch["features_a"]}

        def __len__(self) -> int:
            return len(self.loader)

    adapted_loader = _SingleTrackAdapter(raw_loader)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    trainer = DistillationTrainer(
        teacher=teacher,
        student=student,
        train_loader=adapted_loader,
        lr=args.lr,
        device=args.device,
        weight_decay=args.weight_decay,
    )

    best_metrics = trainer.train(
        num_epochs=args.epochs,
        save_dir=args.output_dir,
    )
    logger.info("Distillation finished. Best metrics: %s", best_metrics)


if __name__ == "__main__":
    main()