from __future__ import annotations

import argparse
from pathlib import Path

from .config import TrainingConfig
from .pipeline import run, run_repeated


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
        device=args.device,
    )
    if args.seeds is not None:
        run_repeated(config, args.seeds)
    else:
        run(config)


if __name__ == "__main__":
    main()
