"""Optuna hyperparameter tuning for the pairwise audio similarity model."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import optuna
from loguru import logger

from src.log_config import setup_logging
from src.models.similarity_head import PairwiseSimilarityModel
from src.models.train import Trainer, build_dataloaders


def _sample_params(trial: optuna.Trial) -> dict:
    return {
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
        "embed_dim": trial.suggest_categorical("embed_dim", [128, 256, 512]),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "contrastive_margin": trial.suggest_float("contrastive_margin", 0.5, 2.0),
        "contrastive_weight": trial.suggest_float("contrastive_weight", 0.1, 1.0),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32, 64]),
        "clip_norm": trial.suggest_float("clip_norm", 0.5, 5.0),
    }


def create_objective(
    train_csv: str,
    val_csv: str,
    feature_dim: int,
    tuning_epochs: int,
    patience: int,
    feature_cache_dir: str | None,
    num_workers: int | None,
    device: str,
    output_dir: str,
) -> Callable[[optuna.Trial], float]:
    """Return an Optuna objective closure that trains one trial."""

    def objective(trial: optuna.Trial) -> float:
        params = _sample_params(trial)

        model = PairwiseSimilarityModel(
            feature_dim=feature_dim,
            embed_dim=params["embed_dim"],
            hidden_dim=params["hidden_dim"],
            dropout=params["dropout"],
        )

        loader_kwargs: dict = {"num_workers": num_workers} if num_workers is not None else {}
        train_loader, val_loader = build_dataloaders(
            train_csv=train_csv,
            val_csv=val_csv,
            batch_size=params["batch_size"],
            feature_cache_dir=feature_cache_dir,
            **loader_kwargs,
        )

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            contrastive_margin=params["contrastive_margin"],
            contrastive_weight=params["contrastive_weight"],
            clip_norm=params["clip_norm"],
            device=device,
            use_mlflow=False,
        )

        save_dir = str(Path(output_dir) / f"trial_{trial.number}")
        metrics = trainer.train(
            num_epochs=tuning_epochs,
            patience=patience,
            save_dir=save_dir,
            trial=trial,
        )

        return metrics.get("val_auc_roc", 0.0)

    return objective


def run_study(args: argparse.Namespace, objective: Callable[[optuna.Trial], float]) -> optuna.Study:
    """Create and run an Optuna study."""
    storage = f"sqlite:///{args.study_db}" if args.study_db else None

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        storage=storage,
        study_name=args.study_name,
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)
    return study


def print_results(study: optuna.Study) -> None:
    """Print best trial results and a retrain CLI command."""
    best = study.best_trial
    logger.info("Best trial: #{} | AUC-ROC: {:.4f}", best.number, best.value)
    logger.info("Best params:")
    for key, value in best.params.items():
        logger.info("  {}: {}", key, value)

    p = best.params
    cmd = (
        "uv run python -m src.models.train"
        f" --pairs_csv <PAIRS_CSV>"
        f" --val_csv <VAL_CSV>"
        f" --embed_dim {p['embed_dim']}"
        f" --hidden_dim {p['hidden_dim']}"
        f" --dropout {p['dropout']}"
        f" --lr {p['lr']}"
        f" --weight_decay {p['weight_decay']}"
        f" --contrastive_margin {p['contrastive_margin']}"
        f" --contrastive_weight {p['contrastive_weight']}"
        f" --clip_norm {p['clip_norm']}"
        f" --batch_size {p['batch_size']}"
    )
    logger.info("Retrain command:\n{}", cmd)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning for similarity model",
    )
    parser.add_argument("--pairs_csv", required=True, help="Training pairs CSV")
    parser.add_argument("--val_csv", required=True, help="Validation pairs CSV")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--timeout", type=int, default=None, help="Total study timeout (seconds)")
    parser.add_argument("--tuning_epochs", type=int, default=20, help="Max epochs per trial")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping per trial")
    parser.add_argument("--study_name", default="orfium-tune", help="Optuna study name")
    parser.add_argument("--study_db", default=None, help="SQLite DB path for persistence")
    parser.add_argument("--output_dir", default="tune_checkpoints", help="Trial checkpoint directory")
    parser.add_argument("--feature_cache_dir", default=None, help="Feature cache location")
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader workers")
    parser.add_argument("--device", default="auto", help="Device")
    parser.add_argument("--feature_dim", type=int, default=452, help="Input feature dimension")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    setup_logging()

    objective = create_objective(
        train_csv=args.pairs_csv,
        val_csv=args.val_csv,
        feature_dim=args.feature_dim,
        tuning_epochs=args.tuning_epochs,
        patience=args.patience,
        feature_cache_dir=args.feature_cache_dir,
        num_workers=args.num_workers,
        device=args.device,
        output_dir=args.output_dir,
    )
    study = run_study(args, objective)
    print_results(study)


if __name__ == "__main__":
    main()