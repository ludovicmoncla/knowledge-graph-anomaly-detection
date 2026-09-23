from __future__ import annotations

import argparse
from pathlib import Path

from .lognet_pipeline import LogNetConfig, run, run_repeated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train LoGNet on a shared prepared experiment")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--prepared-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lognet"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = LogNetConfig(
        data_dir=args.data_dir,
        prepared_data_dir=args.prepared_data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
    )
    if args.seeds is None:
        run(config)
    else:
        run_repeated(config, args.seeds)


if __name__ == "__main__":
    main()
