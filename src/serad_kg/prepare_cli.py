from __future__ import annotations

import argparse
from pathlib import Path

from .preparation import PreparationConfig, prepare_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare shared, reproducible SERAD-KG/LoGNet experiment splits"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", choices=("pooled", "chronological"), required=True)
    parser.add_argument(
        "--negative-sampling", choices=("random", "cache", "genai"), default="random"
    )
    parser.add_argument(
        "--anomaly-cache-dir",
        type=Path,
        help=(
            "Read anomalies from this directory with cache, or save regenerated anomalies "
            "there with genai"
        ),
    )
    parser.add_argument("--openrouter-model", default="openai/gpt-4.1")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--genai-batch-size",
        type=int,
        default=50,
        help="Maximum anomalies requested per OpenRouter call (default: 50)",
    )
    parser.add_argument(
        "--resume-genai",
        action="store_true",
        help="Reuse completed snapshot caches and generate only missing GenAI anomalies",
    )
    parser.add_argument(
        "--num-snapshots",
        "--max-snapshots",
        dest="max_snapshots",
        type=int,
        default=10,
        help="Exact number of chronological snapshots to extract (default: 10)",
    )
    parser.add_argument(
        "--anomalies-per-snapshot",
        type=int,
        default=35,
        help="Legacy fixed anomaly count; ignored when --max-anomaly-ratio is used",
    )
    parser.add_argument(
        "--max-anomaly-ratio",
        type=float,
        help="Build a reusable anomaly pool with this contamination ratio (maximum: 0.20)",
    )
    parser.add_argument(
        "--train-anomaly-ratios",
        type=float,
        nargs="+",
        default=(),
        help=(
            "Create nested training variants while keeping validation/test fixed, "
            "for example: 0.01 0.025 0.05 0.10"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chronological-train-snapshots", type=int, default=7)
    parser.add_argument("--chronological-validation-snapshots", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = PreparationConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        protocol=args.protocol,
        negative_sampling=args.negative_sampling,
        anomaly_cache_dir=args.anomaly_cache_dir,
        max_snapshots=args.max_snapshots,
        anomalies_per_snapshot=args.anomalies_per_snapshot,
        seed=args.seed,
        chronological_train_snapshots=args.chronological_train_snapshots,
        chronological_validation_snapshots=args.chronological_validation_snapshots,
        max_anomaly_ratio=args.max_anomaly_ratio,
        train_anomaly_ratios=tuple(args.train_anomaly_ratios),
        openrouter_model=args.openrouter_model,
        env_file=args.env_file,
        genai_batch_size=args.genai_batch_size,
        resume_genai=args.resume_genai,
    )
    manifest = prepare_experiment(config)
    print(f"Prepared {len(manifest):,} examples in {config.output_dir}")
    print(manifest.groupby(["split", "label"]).size().to_string())
    for ratio in sorted(config.train_anomaly_ratios):
        print(f"Training ratio {ratio:g}: {config.output_dir / f'train_ratio_{ratio:g}'}")


if __name__ == "__main__":
    main()
