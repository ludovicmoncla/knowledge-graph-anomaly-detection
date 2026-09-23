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
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
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
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = TrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        prepared_data_dir=args.prepared_data_dir,
        epochs=args.epochs,
        patience=args.patience,
        anomaly_count=args.anomaly_count,
        negative_sampling=args.negative_sampling,
        anomaly_cache_dir=args.anomaly_cache_dir,
        openrouter_model=args.openrouter_model,
        env_file=args.env_file,
        max_snapshots=args.max_snapshots or None,
        random_seed=args.seed,
        plausibility_weight=args.alpha,
        device=args.device,
    )
    if args.alphas is not None:
        run_alpha_sweep(config, args.alphas, args.seeds)
    elif args.seeds is not None:
        run_repeated(config, args.seeds)
    else:
        run(config)


if __name__ == "__main__":
    main()
