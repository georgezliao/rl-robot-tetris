"""Training command-line entry point."""

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Tetris agent")
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(f"Training loop not implemented; selected config: {args.config}")


if __name__ == "__main__":
    main()
