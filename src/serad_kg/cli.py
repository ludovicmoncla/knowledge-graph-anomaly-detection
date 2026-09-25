from __future__ import annotations

import argparse
from pathlib import Path

from .config import TrainingConfig
from .pipeline import run, run_alpha_sweep, run_repeated


def alpha_value(value: str) -> float:
    alpha = float(value)
    if not 0.0 <= alpha <= 1.0:
        raise argparse.ArgumentTypeError("alpha must be between 0 and 1")
    return alpha


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train SERAD-KG anomaly detection on ICEWS18")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/serad_kg"))
    parser.add_argument(
        "--prepared-data-dir",
        type=Path,
        help="Experiment created by serad-kg-prepare; train one shared static graph",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-5,
        help="Minimum validation-AUPRC increase that resets early stopping",
    )
    parser.add_argument(
        "--edge-mask-ratio",
        type=float,
        default=0.0,
        help="Fraction of positive training edges hidden and used as targets each epoch",
    )
    parser.add_argument("--scheduler-patience", type=int, default=8)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--anomaly-count", type=int, default=30)
    parser.add_argument(
        "--negative-sampling",
        choices=("random", "genai", "cache"),
        default="random",
        help="Strategy used to obtain negative triples",
    )
    parser.add_argument(
        "--anomaly-cache-dir",
        type=Path,
        help="Directory containing shared snapshot_NNN.csv anomaly files",
    )
    parser.add_argument("--openrouter-model", default="openai/gpt-4.1")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--max-snapshots", type=int, default=10,
                        help="Use 0 to process every snapshot")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Run multiple seeds and report mean, standard deviation, and 95%% CI",
    )
    alpha_group = parser.add_mutually_exclusive_group()
    alpha_group.add_argument(
        "--alpha",
        type=alpha_value,
        default=0.5,
        help="Weight of the local plausibility score (between 0 and 1)",
    )
    alpha_group.add_argument(
        "--alphas",
        type=alpha_value,
        nargs="+",
        help="Run the experiment for multiple local-score weights",
    )
    parser.add_argument(
        "--auxiliary-loss-weight",
        type=float,
        default=0.25,
        help="Weight applied to each branch-specific balanced BCE loss",
    )
    parser.add_argument(
        "--ranking-loss-weight",
        type=float,
        default=0.0,
        help="Weight of the differentiable normal-versus-anomaly ranking loss",
    )
    parser.add_argument(
        "--ranking-margin",
        type=float,
        default=0.5,
        help="Desired plausibility-score margin between normal and anomaly pairs",
    )
    parser.add_argument(
        "--score-normalization",
        choices=("stable", "dynamic"),
        default="dynamic",
        help="Use dropout-free stable statistics or training-pass dynamic statistics",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument(
        "--save-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save model.pt after training (use --no-save-model to keep only results)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed seeds whose saved configuration and data still match",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = TrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        prepared_data_dir=args.prepared_data_dir,
        epochs=args.epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        edge_mask_ratio=args.edge_mask_ratio,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        min_learning_rate=args.min_learning_rate,
        anomaly_count=args.anomaly_count,
        negative_sampling=args.negative_sampling,
        anomaly_cache_dir=args.anomaly_cache_dir,
        openrouter_model=args.openrouter_model,
        env_file=args.env_file,
        max_snapshots=args.max_snapshots or None,
        random_seed=args.seed,
        plausibility_weight=args.alpha,
        auxiliary_loss_weight=args.auxiliary_loss_weight,
        ranking_loss_weight=args.ranking_loss_weight,
        ranking_margin=args.ranking_margin,
        score_normalization=args.score_normalization,
        device=args.device,
        save_model=args.save_model,
    )
    if args.alphas is not None:
        run_alpha_sweep(config, args.alphas, args.seeds, resume=args.resume)
    elif args.seeds is not None:
        run_repeated(config, args.seeds, resume=args.resume)
    else:
        run(config)


if __name__ == "__main__":
    main()
