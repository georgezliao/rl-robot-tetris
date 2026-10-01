"""Evaluation command-line entry point."""

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a Tetris agent")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(f"Evaluation loop not implemented; selected config: {args.config}")


if __name__ == "__main__":
    main()
