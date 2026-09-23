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
    parser.add_argument("--negative-sampling", choices=("random", "cache"), default="random")
    parser.add_argument("--anomaly-cache-dir", type=Path)
    parser.add_argument("--max-snapshots", type=int, default=10)
    parser.add_argument("--anomalies-per-snapshot", type=int, default=35)
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
    )
    manifest = prepare_experiment(config)
    print(f"Prepared {len(manifest):,} examples in {config.output_dir}")
    print(manifest.groupby(["split", "label"]).size().to_string())


if __name__ == "__main__":
    main()
