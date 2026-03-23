"""Training pipeline for the pairwise audio similarity model."""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import mlflow
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.features.augmentations import AudioAugmentor
from src.log_config import setup_logging
from src.models.construct_pairs import construct_all_pairs
from src.models.pair_dataset import AudioPairDataset, collate_pairs, compute_feature_stats
from src.models.similarity_head import PairwiseSimilarityModel

torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# Loss utilities
# ---------------------------------------------------------------------------


class FocalLoss(nn.Module):
    """Binary focal loss operating on logits (autocast-safe).

    Focal loss = -alpha * (1 - p_t)^gamma * log(p_t)
    Down-weights easy examples so the model focuses on hard/ambiguous pairs.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * focal_weight * bce).mean()


class ModelEMA:
    """Exponential Moving Average of model weights for stable evaluation."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.decay = decay
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_p, model_p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)

    def state_dict(self) -> dict:
        return self.ema_model.state_dict()


class Trainer:
    """Handles training, validation, and checkpointing for a PairwiseSimilarityModel.

    The *model* must expose ``model.siamese(x1, x2) -> (emb_a, emb_b)`` and
    ``model.head(emb_a, emb_b) -> scores``, matching the interface of
    :class:`~src.models.similarity_head.PairwiseSimilarityModel`.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 1e-4,
        device: str = "auto",
        accumulation_steps: int = 1,
        weight_decay: float = 1e-2,
        scheduler_T_max: int = 50,
        contrastive_margin: float = 0.4,
        contrastive_weight: float = 0.5,
        triplet_weight: float = 0.0,
        triplet_margin: float = 0.3,
        clip_norm: float = 1.0,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        label_smoothing: float = 0.0,
        use_focal: bool = True,
        warmup_epochs: int = 2,
        ema_decay: float = 0.0,
        experiment_name: str = "orfium-similarity",
        tracking_uri: str = "mlruns",
        use_mlflow: bool = True,
    ) -> None:
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.use_mlflow = use_mlflow
        if self.use_mlflow:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)

        self.model = model.to(self.device)

        try:
            self.model = torch.compile(self.model)
        except Exception:
            logger.warning("torch.compile() not supported in this environment; skipping")

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.accumulation_steps = max(accumulation_steps, 1)
        self.contrastive_margin = contrastive_margin
        self.contrastive_weight = contrastive_weight
        self.triplet_weight = triplet_weight
        self.triplet_margin = triplet_margin
        self.clip_norm = clip_norm

        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # Warmup + cosine schedule
        steps_per_epoch = len(train_loader) // max(accumulation_steps, 1)
        total_steps = steps_per_epoch * scheduler_T_max
        warmup_steps = steps_per_epoch * warmup_epochs
        if warmup_steps > 0 and total_steps > warmup_steps:
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=lr,
                total_steps=total_steps,
                pct_start=warmup_steps / total_steps,
                anneal_strategy="cos",
            )
            self._step_scheduler_per_batch = True
        else:
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=scheduler_T_max)
            self._step_scheduler_per_batch = False

        # Loss function: Focal (default) or BCEWithLogits
        if use_focal:
            self.bce_fn = FocalLoss(gamma=focal_gamma, alpha=focal_alpha, label_smoothing=label_smoothing)
            logger.info("Using FocalLoss(γ={}, α={}, smooth={})", focal_gamma, focal_alpha, label_smoothing)
        else:
            self.bce_fn = nn.BCEWithLogitsLoss()
            if label_smoothing > 0:
                logger.warning("Label smoothing requires --focal; ignored with BCEWithLogitsLoss")

        # Triplet loss
        if self.triplet_weight > 0:
            self.triplet_fn = nn.TripletMarginLoss(margin=triplet_margin, p=2)
            logger.info("Triplet loss enabled (weight={}, margin={})", triplet_weight, triplet_margin)

        # EMA
        self.ema: ModelEMA | None = None
        if ema_decay > 0:
            self.ema = ModelEMA(self.model, decay=ema_decay)
            logger.info("EMA enabled (decay={})", ema_decay)

        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------

    def _contrastive_loss(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Cosine contrastive loss: similar pairs -> high cosine sim, dissimilar -> low.

        Uses cosine distance (1 - cosine_similarity) which is scale-invariant and
        better suited for high-dimensional embeddings than Euclidean distance.
        """
        cos_sim = F.cosine_similarity(emb_a, emb_b, dim=-1)
        cos_dist = 1.0 - cos_sim  # range [0, 2]
        pos_loss = labels * cos_dist.pow(2)
        neg_loss = (1.0 - labels) * torch.clamp(self.contrastive_margin - cos_dist, min=0.0).pow(2)
        return (pos_loss + neg_loss).mean()

    @staticmethod
    def contrastive_loss(
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        labels: torch.Tensor,
        margin: float = 0.4,
    ) -> torch.Tensor:
        """Cosine contrastive loss (static, backward-compatible)."""
        cos_sim = F.cosine_similarity(emb_a, emb_b, dim=-1)
        cos_dist = 1.0 - cos_sim
        pos_loss = labels * cos_dist.pow(2)
        neg_loss = (1.0 - labels) * torch.clamp(margin - cos_dist, min=0.0).pow(2)
        return (pos_loss + neg_loss).mean()

    def _triplet_loss(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Online triplet loss with semi-hard mining from the batch.

        For each anchor in the positive set, finds the hardest negative
        (closest dissimilar embedding) within the batch.
        """
        pos_mask = labels > 0.5
        neg_mask = ~pos_mask
        if pos_mask.sum() < 1 or neg_mask.sum() < 1:
            return torch.tensor(0.0, device=emb_a.device)

        # Use emb_a as anchors, emb_b as their pair
        anchors_pos = emb_a[pos_mask]
        positives = emb_b[pos_mask]

        # Pool all negative-pair embeddings as candidate negatives
        neg_pool = torch.cat([emb_a[neg_mask], emb_b[neg_mask]], dim=0)

        # For each positive anchor, find the hardest (closest) negative
        dists = torch.cdist(anchors_pos, neg_pool, p=2)  # (n_pos, n_neg_pool)
        hardest_neg_idx = dists.argmin(dim=1)
        negatives = neg_pool[hardest_neg_idx]

        return self.triplet_fn(anchors_pos, positives, negatives)

    # ------------------------------------------------------------------
    # Training / validation steps
    # ------------------------------------------------------------------

    def train_epoch(self) -> dict[str, float]:
        """Run one training epoch.

        Returns dict with ``loss``, ``bce_loss``, ``contrastive_loss``, ``triplet_loss``, ``accuracy``.
        """
        self.model.train()
        running_loss = 0.0
        running_bce = 0.0
        running_ctr = 0.0
        running_tri = 0.0
        correct = 0
        total = 0

        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(tqdm(self.train_loader, desc="Training", leave=False), 1):
            feat_a = batch["features_a"].to(self.device)
            feat_b = batch["features_b"].to(self.device)
            labels = batch["label"].to(self.device)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                emb_a, emb_b = self.model.siamese(feat_a, feat_b)
                logits = self.model.head(emb_a, emb_b)
                bce = self.bce_fn(logits, labels)

                # Use projected + L2-normalized embeddings for metric losses
                proj_a = F.normalize(self.model.siamese.project(emb_a), dim=-1)
                proj_b = F.normalize(self.model.siamese.project(emb_b), dim=-1)
                ctr = self._contrastive_loss(proj_a, proj_b, labels)
                loss = bce + self.contrastive_weight * ctr

                # Triplet loss with online hard-negative mining
                tri = torch.tensor(0.0, device=self.device)
                if self.triplet_weight > 0:
                    tri = self._triplet_loss(proj_a, proj_b, labels)
                    loss = loss + self.triplet_weight * tri

                loss = loss / self.accumulation_steps

            self.scaler.scale(loss).backward()

            if step % self.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self._step_scheduler_per_batch:
                    self.scheduler.step()
                if self.ema is not None:
                    self.ema.update(self.model)

            bs = labels.size(0)
            running_loss += loss.item() * self.accumulation_steps * bs
            running_bce += bce.item() * bs
            running_ctr += ctr.item() * bs
            running_tri += tri.item() * bs
            correct += ((logits > 0.0).float() == labels).sum().item()
            total += bs

        # Flush any remaining gradients from incomplete accumulation window
        if len(self.train_loader) % self.accumulation_steps != 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

        n = max(total, 1)
        return {
            "loss": running_loss / n,
            "bce_loss": running_bce / n,
            "contrastive_loss": running_ctr / n,
            "triplet_loss": running_tri / n,
            "accuracy": correct / n,
        }

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Run a full validation pass.

        Uses EMA model weights when available for more stable evaluation.
        Returns dict with ``loss``, ``accuracy``, ``auc_roc``.
        """
        eval_model = self.ema.ema_model if self.ema is not None else self.model
        eval_model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_scores: list[float] = []
        all_labels: list[float] = []

        for batch in tqdm(self.val_loader, desc="Validation", leave=False):
            feat_a = batch["features_a"].to(self.device)
            feat_b = batch["features_b"].to(self.device)
            labels = batch["label"].to(self.device)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                emb_a, emb_b = eval_model.siamese(feat_a, feat_b)
                logits = eval_model.head(emb_a, emb_b)
                bce = self.bce_fn(logits, labels)
                # Use projected + L2-normalized embeddings (matching training)
                proj_a = F.normalize(eval_model.siamese.project(emb_a), dim=-1)
                proj_b = F.normalize(eval_model.siamese.project(emb_b), dim=-1)
                ctr = self._contrastive_loss(proj_a, proj_b, labels)
                loss = bce + self.contrastive_weight * ctr

            bs = labels.size(0)
            running_loss += loss.item() * bs
            correct += ((logits > 0.0).float() == labels).sum().item()
            total += bs

            all_scores.extend(torch.sigmoid(logits).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        n = max(total, 1)
        auc = 0.0
        if len(set(all_labels)) > 1:
            auc = float(roc_auc_score(all_labels, all_scores))

        return {
            "loss": running_loss / n,
            "accuracy": correct / n,
            "auc_roc": auc,
        }

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(
        self,
        num_epochs: int = 50,
        patience: int = 10,
        save_dir: str = "models",
        trial: object | None = None,
    ) -> dict[str, float]:
        """Full training loop with early stopping. Returns best metrics."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        if self.use_mlflow:
            mlflow.start_run()
            mlflow.log_params({
                "lr": self.optimizer.param_groups[0]["lr"],
                "weight_decay": self.optimizer.param_groups[0]["weight_decay"],
                "accumulation_steps": self.accumulation_steps,
                "num_epochs": num_epochs,
                "patience": patience,
                "device": str(self.device),
            })
            if self.device.type == "cuda":
                mlflow.log_param("gpu_name", torch.cuda.get_device_name(self.device))

        best_auc = 0.0
        best_epoch = 0
        no_improve = 0
        best_metrics: dict[str, float] = {}

        try:
            for epoch in range(1, num_epochs + 1):
                train_m = self.train_epoch()
                val_m = self.validate()
                if not self._step_scheduler_per_batch:
                    self.scheduler.step()

                if self.use_mlflow:
                    mlflow.log_metrics({
                        "train_loss": train_m["loss"],
                        "train_accuracy": train_m["accuracy"],
                        "train_bce_loss": train_m["bce_loss"],
                        "train_contrastive_loss": train_m["contrastive_loss"],
                        "train_triplet_loss": train_m["triplet_loss"],
                        "val_loss": val_m["loss"],
                        "val_accuracy": val_m["accuracy"],
                        "val_auc_roc": val_m["auc_roc"],
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                    }, step=epoch)

                if trial is not None:
                    trial.report(val_m["auc_roc"], epoch)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

                is_best = val_m["auc_roc"] > best_auc
                if is_best:
                    best_auc = val_m["auc_roc"]
                    best_epoch = epoch
                    best_metrics = {
                        **train_m,
                        **{f"val_{k}": v for k, v in val_m.items()},
                    }
                    no_improve = 0
                    ckpt = {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "metrics": best_metrics,
                    }
                    if self.ema is not None:
                        ckpt["ema_state_dict"] = self.ema.state_dict()
                    torch.save(
                        ckpt,
                        save_path / "best_model.pt",
                    )
                    if self.use_mlflow:
                        mlflow.log_artifact(str(save_path / "best_model.pt"))
                        mlflow.log_metric("best_auc", best_auc, step=epoch)
                        mlflow.set_tag("best_epoch", str(best_epoch))
                else:
                    no_improve += 1

                marker = " *" if is_best else ""
                logger.info(
                    "Epoch {:3d}/{} | train_loss={:.4f} | val_loss={:.4f} | "
                    "val_auc={:.4f} | best={:.4f} (ep{}){}"
                    .format(
                        epoch,
                        num_epochs,
                        train_m["loss"],
                        val_m["loss"],
                        val_m["auc_roc"],
                        best_auc,
                        best_epoch,
                        marker,
                    )
                )

                if no_improve >= patience:
                    logger.info(
                        "Early stopping at epoch {} (no improvement for {} epochs)",
                        epoch,
                        patience,
                    )
                    break
        finally:
            if self.use_mlflow:
                mlflow.end_run()

        return best_metrics


def build_dataloaders(
    train_csv: str,
    val_csv: str,
    batch_size: int = 32,
    num_workers: int | None = None,
    feature_cache_dir: str | None = None,
    train_augmentor: object | None = None,
    **dataset_kwargs,
) -> tuple[DataLoader, DataLoader]:
    """Create optimized DataLoaders for training and validation."""
    if num_workers is None:
        num_workers = min(8, os.cpu_count() or 4)

    train_ds = AudioPairDataset(
        train_csv, feature_cache_dir=feature_cache_dir, augmentor=train_augmentor,
        training=True, **dataset_kwargs,
    )
    val_ds = AudioPairDataset(
        val_csv, feature_cache_dir=feature_cache_dir, training=False, **dataset_kwargs,
    )

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=collate_pairs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=collate_pairs,
    )
    return train_loader, val_loader


def build_training_pairs(
    data_dir: str,
    output_csv: str,
    val_split: float = 0.15,
    test_split: float = 0.15,
) -> None:
    """Scan data directories and construct training pair CSV files.

    Delegates to :func:`src.models.construct_pairs.construct_all_pairs`.

    Parameters
    ----------
    data_dir:
        Root directory containing ``sonics/``, ``mippia/``, ``fakemusiccaps/``.
    output_csv:
        Directory where ``train.csv``, ``val.csv``, ``test.csv`` will be written.
    val_split:
        Fraction of pairs reserved for validation.
    test_split:
        Fraction of pairs reserved for testing.
    """
    construct_all_pairs(
        data_dir=data_dir,
        output_dir=output_csv,
        val_split=val_split,
        test_split=test_split,
    )


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or evaluate the pairwise audio similarity model",
    )
    parser.add_argument("--pairs_csv", required=True, help="Path to training pairs CSV")
    parser.add_argument("--val_csv", default=None, help="Path to validation pairs CSV (inferred from pairs_csv dir if not set)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--output_dir", default="models", help="Directory for saving checkpoints")
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader workers (default: min(8, cpu_count))")
    parser.add_argument("--feature_cache_dir", default=None, help="Directory for caching extracted features")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Similarity head hidden dim")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="AdamW weight decay")
    parser.add_argument("--contrastive_margin", type=float, default=0.4, help="Cosine contrastive loss margin (default 0.4 for cosine distance)")
    parser.add_argument("--contrastive_weight", type=float, default=0.5, help="Contrastive loss weight")
    parser.add_argument("--clip_norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True, help="Enable training-time audio augmentation (--no-augment to disable)")
    parser.add_argument("--aug_prob", type=float, default=0.3, help="Per-augmentation probability")
    parser.add_argument("--n_augmentations", type=int, default=0, help="Number of pre-computed augmented variants (0=live augmentation, >0=use cached variants)")
    parser.add_argument("--feature_noise_std", type=float, default=0.01, help="Gaussian noise std for feature-space perturbation (0=disabled)")
    parser.add_argument("--feature_dropout_p", type=float, default=0.05, help="Feature dropout probability (0=disabled)")
    parser.add_argument("--eval_only", action="store_true", help="Run evaluation only (requires --model_path)")
    parser.add_argument("--model_path", default=None, help="Path to model checkpoint (for --eval_only)")
    parser.add_argument("--experiment_name", default="orfium-similarity", help="MLflow experiment name")
    parser.add_argument("--run_name", default=None, help="MLflow run name (default: auto)")
    parser.add_argument("--mlflow_tracking_uri", default="mlruns", help="MLflow tracking URI")
    parser.add_argument("--no_mlflow", action="store_true", help="Disable MLflow tracking")
    parser.add_argument(
        "--dual-stream", action="store_true", help="Use dual-stream encoder (Phase 3)"
    )
    parser.add_argument(
        "--use-segment-transformer",
        action="store_true",
        help="Use SegmentTransformer instead of AttentionPooling (Phase 4)",
    )
    parser.add_argument("--triplet-weight", type=float, default=0.0, help="Triplet loss weight (0=disabled)")
    parser.add_argument("--triplet-margin", type=float, default=0.3, help="Triplet loss margin")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma (focusing parameter)")
    parser.add_argument("--focal-alpha", type=float, default=0.25, help="Focal loss alpha (class balance)")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing factor (0=disabled)")
    parser.add_argument("--no-focal", action="store_true", help="Use BCEWithLogitsLoss instead of FocalLoss")
    parser.add_argument("--warmup-epochs", type=int, default=2, help="Linear warmup epochs before cosine annealing")
    parser.add_argument("--ema-decay", type=float, default=0.0, help="EMA decay rate (0=disabled, try 0.999)")
    parser.add_argument(
        "--load-pretrained", default=None, help="Path to pre-trained checkpoint to initialize from"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    setup_logging()

    if args.eval_only:
        if args.model_path is None:
            raise SystemExit("--eval_only requires --model_path")

        checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        # torch.compile() prefixes keys with "_orig_mod." — strip it for portability
        if any(k.startswith("_orig_mod.") for k in state_dict):
            state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}

        proj_key = "siamese.projector.net.0.weight"
        feature_dim = state_dict[proj_key].shape[1] if proj_key in state_dict else 452

        model = PairwiseSimilarityModel(feature_dim=feature_dim)
        model.load_state_dict(state_dict)

        # Feature standardization must match training (use same stats)
        eval_kwargs: dict = {}
        if args.feature_cache_dir:
            stats_file = Path(args.feature_cache_dir) / "feature_stats.npz"
            if stats_file.exists():
                eval_kwargs["feature_stats_path"] = str(stats_file)

        _, eval_loader = build_dataloaders(
            train_csv=args.pairs_csv,
            val_csv=args.pairs_csv,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            feature_cache_dir=args.feature_cache_dir,
            **eval_kwargs,
        )

        device = args.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dev = torch.device(device)
        model.to(dev)

        trainer = Trainer.__new__(Trainer)
        trainer.device = dev
        trainer.model = model
        trainer.val_loader = eval_loader
        trainer.bce_fn = FocalLoss() if not args.no_focal else nn.BCEWithLogitsLoss()
        trainer.ema = None
        trainer.use_amp = dev.type == "cuda"
        trainer.scaler = torch.amp.GradScaler("cuda", enabled=trainer.use_amp)
        trainer.contrastive_margin = args.contrastive_margin
        trainer.contrastive_weight = args.contrastive_weight

        metrics = trainer.validate()
        logger.info("Evaluation results: {}", metrics)
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        if not args.no_mlflow:
            mlflow.set_tracking_uri(args.mlflow_tracking_uri)
            mlflow.set_experiment(args.experiment_name)
            with mlflow.start_run(run_name=args.run_name or "eval"):
                mlflow.log_param("eval_only", True)
                mlflow.log_param("model_path", args.model_path)
                mlflow.log_metrics(metrics)

        return

    # --- Training mode ---
    val_csv = args.val_csv
    if val_csv is None:
        pairs_dir = Path(args.pairs_csv).parent
        val_csv = str(pairs_dir / "val_pairs.csv")

    train_augmentor = None
    train_dataset_kwargs: dict = {}

    if args.n_augmentations > 0:
        # Hybrid mode: use pre-computed augmented variants + feature-space perturbation
        train_dataset_kwargs["n_augmentations"] = args.n_augmentations
        train_dataset_kwargs["feature_noise_std"] = args.feature_noise_std
        train_dataset_kwargs["feature_dropout_p"] = args.feature_dropout_p
        logger.info(
            "Hybrid augmentation: {} cached variants + feature noise(σ={}) + dropout(p={})",
            args.n_augmentations, args.feature_noise_std, args.feature_dropout_p,
        )
    elif args.augment:
        # Legacy mode: live audio augmentation (slower, bypasses feature cache)
        train_augmentor = AudioAugmentor(sr=16000, p=args.aug_prob, enabled=True)
        logger.info("Live audio augmentation enabled (p={})", args.aug_prob)

    # Auto-compute feature standardization stats from training cache
    feature_stats_path = None
    if args.feature_cache_dir:
        stats_file = Path(args.feature_cache_dir) / "feature_stats.npz"
        if not stats_file.exists():
            logger.info("Computing feature statistics from cache...")
            compute_feature_stats(args.feature_cache_dir)
        feature_stats_path = str(stats_file)
        train_dataset_kwargs["feature_stats_path"] = feature_stats_path

    train_loader, val_loader = build_dataloaders(
        train_csv=args.pairs_csv,
        val_csv=val_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        feature_cache_dir=args.feature_cache_dir,
        train_augmentor=train_augmentor,
        **train_dataset_kwargs,
    )

    # Probe feature dimension from the first training sample
    sample = next(iter(train_loader))
    feature_dim = sample["features_a"].shape[-1]
    logger.info("Detected feature_dim={}", feature_dim)

    # Build the model, wiring experimental architecture flags
    use_spectrogram = getattr(args, "dual_stream", False)
    use_segment_transformer = getattr(args, "use_segment_transformer", False)

    model = PairwiseSimilarityModel(
        feature_dim=feature_dim,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        use_spectrogram=use_spectrogram,
        use_segment_transformer=use_segment_transformer,
    )

    # Optionally load pre-trained weights before training
    if args.load_pretrained:
        pretrained_ckpt = torch.load(args.load_pretrained, map_location="cpu", weights_only=False)
        pretrained_sd = pretrained_ckpt.get("model_state_dict", pretrained_ckpt)
        if any(k.startswith("_orig_mod.") for k in pretrained_sd):
            pretrained_sd = {k.removeprefix("_orig_mod."): v for k, v in pretrained_sd.items()}
        missing, unexpected = model.load_state_dict(pretrained_sd, strict=False)
        logger.info("Loaded pre-trained weights from {}", args.load_pretrained)
        if missing:
            logger.warning("Missing keys in pre-trained checkpoint: {}", missing)
        if unexpected:
            logger.warning("Unexpected keys in pre-trained checkpoint: {}", unexpected)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        device=args.device,
        accumulation_steps=args.accumulation_steps,
        weight_decay=args.weight_decay,
        scheduler_T_max=args.epochs,
        contrastive_margin=args.contrastive_margin,
        contrastive_weight=args.contrastive_weight,
        triplet_weight=args.triplet_weight,
        triplet_margin=args.triplet_margin,
        clip_norm=args.clip_norm,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        label_smoothing=args.label_smoothing,
        use_focal=not args.no_focal,
        warmup_epochs=args.warmup_epochs,
        ema_decay=args.ema_decay,
        experiment_name=args.experiment_name,
        tracking_uri=args.mlflow_tracking_uri,
        use_mlflow=not args.no_mlflow,
    )

    best_metrics = trainer.train(
        num_epochs=args.epochs,
        patience=args.patience,
        save_dir=args.output_dir,
    )
    logger.info("Training complete. Best metrics: {}", best_metrics)


if __name__ == "__main__":
    main()
