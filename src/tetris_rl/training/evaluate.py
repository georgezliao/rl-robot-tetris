"""Deterministic held-out evaluation for visual-policy checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch import Tensor

from tetris_rl.envs.javafx import JavaFxControlEnv
from tetris_rl.events.observations import observation_sha256
from tetris_rl.events.recorder import EpisodeRecorder
from tetris_rl.events.schema import EpisodeManifest, JsonScalar, StepEvent
from tetris_rl.models.cnn_policy import VisualActorCritic
from tetris_rl.training.config import VisualExperimentConfig, load_visual_config
from tetris_rl.training.visual import (
    action_set,
    resolve_device,
    validate_runtime_support,
)
from tetris_rl.types import ControlAction, Observation, StepResult

ENVIRONMENT_VERSION = "tetris-javafx-visual-v1"
CHECKPOINT_SCHEMA_VERSION = 1
COMPATIBILITY_SECTIONS = ("observation", "actions", "model", "reward")


class CheckpointCompatibilityError(ValueError):
    """Raised when a checkpoint cannot represent the requested condition."""


class EvaluationEnvironment(Protocol):
    def reset(self, seed: int | None = None) -> Observation: ...

    def step(self, action: ControlAction | int) -> StepResult: ...


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    model: VisualActorCritic
    sha256: str
    environment_steps: int
    action_labels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    episode_index: int
    episode_id: str
    seed: int
    steps: int
    terminated: bool
    truncated: bool
    episode_return: float
    score: int
    lines: int
    pieces: int
    recording_directory: str


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    run_directory: Path
    episodes: tuple[EpisodeResult, ...]
    summary: dict[str, Any]


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_state_sha256(model: VisualActorCritic) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(",".join(map(str, tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CheckpointCompatibilityError(f"checkpoint {name} must be a mapping")
    return value


def _check_compatibility(
    saved_config: Mapping[str, Any], config: VisualExperimentConfig
) -> None:
    requested = config.to_dict()
    for section in COMPATIBILITY_SECTIONS:
        if section not in saved_config:
            raise CheckpointCompatibilityError(
                f"checkpoint configuration is missing {section!r}"
            )
        if saved_config[section] != requested[section]:
            raise CheckpointCompatibilityError(
                f"checkpoint {section} does not match evaluation configuration: "
                f"{saved_config[section]!r} != {requested[section]!r}"
            )


def load_visual_checkpoint(
    checkpoint_path: Path | str,
    config: VisualExperimentConfig,
) -> LoadedCheckpoint:
    """Validate and reconstruct the policy encoded by a training checkpoint."""
    validate_runtime_support(config)
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exception:
        raise CheckpointCompatibilityError(
            f"Could not load checkpoint {path}: {exception}"
        ) from exception
    checkpoint = _mapping(checkpoint, "root")
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointCompatibilityError(
            "unsupported checkpoint schema version "
            f"{checkpoint.get('schema_version')!r}"
        )
    saved_config = _mapping(checkpoint.get("config"), "config")
    _check_compatibility(saved_config, config)

    environment_steps = checkpoint.get("environment_steps")
    if (
        isinstance(environment_steps, bool)
        or not isinstance(environment_steps, int)
        or environment_steps < 0
    ):
        raise CheckpointCompatibilityError(
            "checkpoint environment_steps must be a non-negative integer"
        )
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise CheckpointCompatibilityError(
            "checkpoint model_state_dict must be a mapping"
        )

    actions = action_set(config)
    model = VisualActorCritic(
        config.observation.shape,
        len(actions),
        config.model.feature_dim,
    )
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exception:
        raise CheckpointCompatibilityError(
            f"checkpoint parameters do not match the configured model: {exception}"
        ) from exception
    device = resolve_device(config.training.device)
    model.to(device)
    model.eval()
    return LoadedCheckpoint(
        model=model,
        sha256=_checkpoint_sha256(path),
        environment_steps=environment_steps,
        action_labels=tuple(action.name.lower() for action in actions),
    )


def _validate_observation(
    observation: np.ndarray, expected_shape: tuple[int, int, int]
) -> None:
    if observation.shape != expected_shape:
        raise RuntimeError(
            f"Expected observation shape {expected_shape}, got {observation.shape}"
        )
    if observation.dtype != np.float32:
        raise RuntimeError(f"Expected float32 observation, got {observation.dtype}")
    if not np.isfinite(observation).all():
        raise RuntimeError("Observation contains non-finite values")
    if float(observation.min()) < 0.0 or float(observation.max()) > 1.0:
        raise RuntimeError("Observation values must lie in [0, 1]")


def _tensor_observation(observation: np.ndarray, device: torch.device) -> Tensor:
    return torch.from_numpy(np.ascontiguousarray(observation)).unsqueeze(0).to(device)


def _scalar_metrics(info: Mapping[str, object]) -> dict[str, JsonScalar]:
    metrics: dict[str, JsonScalar] = {}
    for key, value in info.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"Metric {key!r} must be finite")
            metrics[key] = value
        else:
            raise TypeError(f"Metric {key!r} is not a JSON scalar")
    return metrics


def _integer_metric(info: Mapping[str, object], name: str) -> int:
    value = info.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Expected integer metric {name!r}, got {value!r}")
    return value


def _write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _aggregate(
    episodes: Sequence[EpisodeResult],
    *,
    loaded: LoadedCheckpoint,
    seeds: Sequence[int],
    max_steps_per_episode: int,
    parameters_unchanged: bool,
) -> dict[str, Any]:
    total_lines = sum(episode.lines for episode in episodes)
    total_score = sum(episode.score for episode in episodes)
    total_pieces = sum(episode.pieces for episode in episodes)
    episode_count = len(episodes)
    return {
        "checkpoint_sha256": loaded.sha256,
        "checkpoint_environment_steps": loaded.environment_steps,
        "action_labels": list(loaded.action_labels),
        "seeds": list(seeds),
        "max_steps_per_episode": max_steps_per_episode,
        "episode_count": episode_count,
        "terminated_episodes": sum(episode.terminated for episode in episodes),
        "truncated_episodes": sum(episode.truncated for episode in episodes),
        "total_environment_steps": sum(episode.steps for episode in episodes),
        "total_lines": total_lines,
        "total_score": total_score,
        "total_pieces": total_pieces,
        "mean_lines": total_lines / episode_count,
        "mean_score": total_score / episode_count,
        "mean_survival_pieces": total_pieces / episode_count,
        "lines_per_100_pieces": (
            None if total_pieces == 0 else 100.0 * total_lines / total_pieces
        ),
        "parameters_unchanged": parameters_unchanged,
        "deterministic_action_selection": True,
    }


def evaluate_loaded_policy(
    config: VisualExperimentConfig,
    loaded: LoadedCheckpoint,
    environment: EvaluationEnvironment,
    *,
    output_root: Path,
    episode_count: int,
    max_steps_per_episode: int,
) -> EvaluationResult:
    """Evaluate an already loaded policy and write generic episode records."""
    if episode_count < 1 or max_steps_per_episode < 1:
        raise ValueError("episode_count and max_steps_per_episode must be positive")
    actions = action_set(config)
    expected_labels = tuple(action.name.lower() for action in actions)
    if loaded.action_labels != expected_labels:
        raise CheckpointCompatibilityError("loaded action labels are inconsistent")

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    run_directory = output_root / f"{config.name}-{run_id}"
    episode_root = run_directory / "episodes"
    episode_root.mkdir(parents=True)
    seeds = tuple(
        config.evaluation.first_seed + index for index in range(episode_count)
    )
    device = next(loaded.model.parameters()).device
    parameter_digest_before = _model_state_sha256(loaded.model)
    results: list[EpisodeResult] = []

    loaded.model.eval()
    for episode_index, seed in enumerate(seeds):
        manifest = EpisodeManifest.create(
            game_id="tetris",
            environment_version=ENVIRONMENT_VERSION,
            seed=seed,
            action_spec=loaded.action_labels,
            observation_spec={
                "kind": "stacked_rendered_rgb",
                "shape": "x".join(map(str, config.observation.shape)),
                "dtype": "float32",
                "minimum": 0.0,
                "maximum": 1.0,
            },
            configuration={
                "experiment": config.to_dict(),
                "evaluation_episode_index": episode_index,
                "max_steps_per_episode": max_steps_per_episode,
                "deterministic_action_selection": True,
                "checkpoint_environment_steps": loaded.environment_steps,
            },
            model_id=f"sha256:{loaded.sha256}",
        )
        recording_directory = episode_root / (
            f"episode-{episode_index:04d}-seed-{seed}-{manifest.episode_id}"
        )
        observation = environment.reset(seed=seed)
        _validate_observation(observation, config.observation.shape)
        cumulative_reward = 0.0
        done = False
        final_info: Mapping[str, object] = {}

        with EpisodeRecorder(recording_directory, manifest) as recorder:
            for step_index in range(max_steps_per_episode):
                observation_tensor = _tensor_observation(observation, device)
                with torch.inference_mode():
                    distribution, value = loaded.model.distribution_and_value(
                        observation_tensor
                    )
                    action_tensor = distribution.logits.argmax(dim=-1)
                    action_index = int(action_tensor.item())
                    action_probability = float(
                        distribution.probs[0, action_index].item()
                    )
                    value_estimate = float(value.item())

                observation, raw_reward, done, final_info = environment.step(
                    actions[action_index]
                )
                _validate_observation(observation, config.observation.shape)
                reward = raw_reward + (config.reward.terminal_penalty if done else 0.0)
                cumulative_reward += reward
                recorder.record(
                    StepEvent(
                        episode_id=manifest.episode_id,
                        step_index=step_index,
                        action=action_index,
                        action_label=loaded.action_labels[action_index],
                        reward=reward,
                        cumulative_reward=cumulative_reward,
                        done=done,
                        metrics=_scalar_metrics(final_info),
                        observation_digest=observation_sha256(observation),
                        model_output={
                            "selected_action_probability": action_probability,
                            "value_estimate": value_estimate,
                        },
                    )
                )
                if done:
                    break

        episode_result = EpisodeResult(
            episode_index=episode_index,
            episode_id=manifest.episode_id,
            seed=seed,
            steps=step_index + 1,
            terminated=done,
            truncated=not done,
            episode_return=cumulative_reward,
            score=_integer_metric(final_info, "score"),
            lines=_integer_metric(final_info, "total_lines"),
            pieces=_integer_metric(final_info, "pieces_placed"),
            recording_directory=str(recording_directory),
        )
        results.append(episode_result)
        _write_json(recording_directory / "result.json", asdict(episode_result))

    parameter_digest_after = _model_state_sha256(loaded.model)
    parameters_unchanged = parameter_digest_before == parameter_digest_after
    if not parameters_unchanged:
        raise RuntimeError("Model parameters changed during evaluation")
    summary = _aggregate(
        results,
        loaded=loaded,
        seeds=seeds,
        max_steps_per_episode=max_steps_per_episode,
        parameters_unchanged=parameters_unchanged,
    )
    _write_json(run_directory / "episodes.json", [asdict(item) for item in results])
    _write_json(run_directory / "summary.json", summary)
    return EvaluationResult(run_directory, tuple(results), summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        help="override evaluation.output_directory",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        help="bounded diagnostic override; seeds still start at evaluation.first_seed",
    )
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        help="bounded diagnostic override",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_visual_config(args.config)
    episode_count = (
        config.evaluation.episodes if args.episodes is None else args.episodes
    )
    max_steps = (
        config.evaluation.max_steps_per_episode
        if args.max_steps_per_episode is None
        else args.max_steps_per_episode
    )
    if episode_count < 1 or max_steps < 1:
        raise SystemExit("episode and step limits must be positive")
    loaded = load_visual_checkpoint(args.checkpoint, config)
    output_root = args.output_root or config.evaluation.output_directory
    with JavaFxControlEnv(
        observation_size=(config.observation.height, config.observation.width),
        frame_stack=config.observation.frame_stack,
    ) as environment:
        result = evaluate_loaded_policy(
            config,
            loaded,
            environment,
            output_root=output_root,
            episode_count=episode_count,
            max_steps_per_episode=max_steps,
        )
    print(
        json.dumps(
            {"run_directory": str(result.run_directory), **result.summary},
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
