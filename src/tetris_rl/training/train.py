"""Training command-line entry point."""

import argparse
import json
from pathlib import Path

from tetris_rl.training.config import load_visual_config
from tetris_rl.training.visual import experiment_summary, train_visual_agent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Tetris agent")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate configuration and model shapes without starting JavaFX",
    )
    parser.add_argument(
        "--environment-steps",
        type=int,
        help="override the configured step count for a bounded diagnostic run",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_visual_config(args.config)
    if args.dry_run:
        if args.environment_steps is not None:
            raise SystemExit("--environment-steps has no effect with --dry-run")
        print(json.dumps(experiment_summary(config), indent=2, sort_keys=True))
        return
    result = train_visual_agent(
        config,
        environment_steps_override=args.environment_steps,
        config_path=args.config,
    )
    print(
        json.dumps(
            {
                "run_directory": str(result.run_directory),
                "environment_steps": result.environment_steps,
                "completed_episodes": result.completed_episodes,
                "updates": result.updates,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
