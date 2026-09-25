from __future__ import annotations

import argparse
from pathlib import Path

from .lognet_pipeline import LogNetConfig, run, run_repeated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train LoGNet on a shared prepared experiment")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--prepared-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lognet"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--device", default="auto")
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
    config = LogNetConfig(
        data_dir=args.data_dir,
        prepared_data_dir=args.prepared_data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        seed=args.seed,
        device=args.device,
        save_model=args.save_model,
    )
    if args.seeds is None:
        run(config)
    else:
        run_repeated(config, args.seeds, resume=args.resume)


if __name__ == "__main__":
    main()
