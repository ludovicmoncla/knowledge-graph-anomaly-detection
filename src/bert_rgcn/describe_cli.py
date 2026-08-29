from __future__ import annotations

import argparse
import json
from pathlib import Path

from .description import describe_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate ICEWS18 and generate descriptive dataset artifacts"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/icews18"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/icews18"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = describe_dataset(args.data_dir, args.output_dir)
    print(json.dumps(summary, indent=2))
    print(f"Derived files written to {args.output_dir}")


if __name__ == "__main__":
    main()
