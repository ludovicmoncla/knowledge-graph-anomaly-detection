from __future__ import annotations

import argparse
from pathlib import Path

from .config import TrainingConfig
from .pipeline import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train BERT + R-GCN anomaly detection on ICEWS18")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bert_rgcn"))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--anomaly-count", type=int, default=30)
    parser.add_argument(
        "--negative-sampling",
        choices=("random", "genai"),
        default="random",
        help="Strategy used to generate negative triples",
    )
    parser.add_argument("--openrouter-model", default="openai/gpt-4.1")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--max-snapshots", type=int, default=10,
                        help="Use 0 to process every snapshot")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = TrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        patience=args.patience,
        anomaly_count=args.anomaly_count,
        negative_sampling=args.negative_sampling,
        openrouter_model=args.openrouter_model,
        env_file=args.env_file,
        max_snapshots=args.max_snapshots or None,
        random_seed=args.seed,
        device=args.device,
    )
    run(config)


if __name__ == "__main__":
    main()
