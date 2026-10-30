"""Typed loading and validation for visual-agent experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """Raised when an experiment configuration is incomplete or inconsistent."""


def _section(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{name} must be a mapping")
    return dict(value)


def _check_keys(
    value: dict[str, Any],
    name: str,
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    missing = required - value.keys()
    unknown = value.keys() - required - optional
    if missing:
        raise ConfigurationError(f"{name} is missing keys: {sorted(missing)}")
    if unknown:
        raise ConfigurationError(f"{name} has unknown keys: {sorted(unknown)}")


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigurationError(f"{name} must be a positive integer")
    return value


def _probability(value: Any, name: str, *, include_one: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be numeric")
    number = float(value)
    upper_valid = number <= 1.0 if include_one else number < 1.0
    if number < 0.0 or not upper_valid:
        interval = "[0, 1]" if include_one else "[0, 1)"
        raise ConfigurationError(f"{name} must lie in {interval}")
    return number


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be numeric")
    number = float(value)
    if number <= 0.0:
        raise ConfigurationError(f"{name} must be positive")
    return number


@dataclass(frozen=True, slots=True)
class ObservationConfig:
    region: str
    color: str
    height: int
    width: int
    frame_stack: int

    @property
    def channels(self) -> int:
        channels_per_frame = 3 if self.color == "rgb" else 1
        return channels_per_frame * self.frame_stack

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.channels, self.height, self.width


@dataclass(frozen=True, slots=True)
class ActionConfig:
    gravity_ticks_per_step: int
    allow_hold: bool
    macro_actions: bool


@dataclass(frozen=True, slots=True)
class ModelConfig:
    architecture: str
    feature_dim: int


@dataclass(frozen=True, slots=True)
class AlgorithmConfig:
    kind: str
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_epsilon: float
    value_coefficient: float
    entropy_coefficient: float
    rollout_steps: int
    update_epochs: int
    minibatch_size: int
    max_gradient_norm: float


@dataclass(frozen=True, slots=True)
class RewardConfig:
    kind: str
    terminal_penalty: float


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    environment_steps: int
    checkpoint_interval: int
    output_directory: Path
    device: str


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    episodes: int
    first_seed: int
    max_steps_per_episode: int
    output_directory: Path


@dataclass(frozen=True, slots=True)
class VisualExperimentConfig:
    name: str
    seed: int
    observation: ObservationConfig
    actions: ActionConfig
    model: ModelConfig
    algorithm: AlgorithmConfig
    reward: RewardConfig
    training: TrainingConfig
    evaluation: EvaluationConfig

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["training"]["output_directory"] = str(self.training.output_directory)
        value["evaluation"]["output_directory"] = str(self.evaluation.output_directory)
        return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_mapping(path: Path, active: tuple[Path, ...] = ()) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in active:
        chain = " -> ".join(str(item) for item in (*active, resolved))
        raise ConfigurationError(f"Configuration inheritance cycle: {chain}")
    if not resolved.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {resolved}")
    with resolved.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    value = _section(loaded, str(resolved))
    extends = value.pop("extends", None)
    if extends is None:
        return value
    if not isinstance(extends, str) or not extends:
        raise ConfigurationError("extends must be a non-empty relative path")
    parent = _load_mapping(resolved.parent / extends, (*active, resolved))
    return _deep_merge(parent, value)


def load_visual_config(path: Path | str) -> VisualExperimentConfig:
    """Load a strict visual experiment configuration from YAML."""
    raw = _load_mapping(Path(path))
    _check_keys(
        raw,
        "root",
        required={
            "name",
            "seed",
            "observation",
            "actions",
            "model",
            "algorithm",
            "reward",
            "training",
            "evaluation",
        },
    )
    if not isinstance(raw["name"], str) or not raw["name"]:
        raise ConfigurationError("name must be a non-empty string")
    if isinstance(raw["seed"], bool) or not isinstance(raw["seed"], int):
        raise ConfigurationError("seed must be an integer")

    observation_raw = _section(raw["observation"], "observation")
    _check_keys(
        observation_raw,
        "observation",
        required={"region", "color", "height", "width", "frame_stack"},
    )
    if observation_raw["region"] not in {"full_ui", "board", "board_previews"}:
        raise ConfigurationError("observation.region is not supported")
    if observation_raw["color"] not in {"rgb", "grayscale"}:
        raise ConfigurationError("observation.color must be rgb or grayscale")
    observation = ObservationConfig(
        region=observation_raw["region"],
        color=observation_raw["color"],
        height=_positive_int(observation_raw["height"], "observation.height"),
        width=_positive_int(observation_raw["width"], "observation.width"),
        frame_stack=_positive_int(
            observation_raw["frame_stack"], "observation.frame_stack"
        ),
    )

    actions_raw = _section(raw["actions"], "actions")
    _check_keys(
        actions_raw,
        "actions",
        required={"gravity_ticks_per_step", "allow_hold", "macro_actions"},
    )
    if not isinstance(actions_raw["allow_hold"], bool) or not isinstance(
        actions_raw["macro_actions"], bool
    ):
        raise ConfigurationError("actions boolean fields must be true or false")
    actions = ActionConfig(
        gravity_ticks_per_step=_positive_int(
            actions_raw["gravity_ticks_per_step"],
            "actions.gravity_ticks_per_step",
        ),
        allow_hold=actions_raw["allow_hold"],
        macro_actions=actions_raw["macro_actions"],
    )

    model_raw = _section(raw["model"], "model")
    _check_keys(model_raw, "model", required={"architecture", "feature_dim"})
    if model_raw["architecture"] != "small_cnn":
        raise ConfigurationError("Only model.architecture=small_cnn is implemented")
    model = ModelConfig(
        architecture=model_raw["architecture"],
        feature_dim=_positive_int(model_raw["feature_dim"], "model.feature_dim"),
    )

    algorithm_raw = _section(raw["algorithm"], "algorithm")
    _check_keys(
        algorithm_raw,
        "algorithm",
        required={
            "kind",
            "learning_rate",
            "gamma",
            "gae_lambda",
            "clip_epsilon",
            "value_coefficient",
            "entropy_coefficient",
            "rollout_steps",
            "update_epochs",
            "minibatch_size",
            "max_gradient_norm",
        },
    )
    if algorithm_raw["kind"] != "ppo":
        raise ConfigurationError("Only algorithm.kind=ppo is implemented")
    algorithm = AlgorithmConfig(
        kind=algorithm_raw["kind"],
        learning_rate=_positive_float(
            algorithm_raw["learning_rate"], "algorithm.learning_rate"
        ),
        gamma=_probability(algorithm_raw["gamma"], "algorithm.gamma"),
        gae_lambda=_probability(algorithm_raw["gae_lambda"], "algorithm.gae_lambda"),
        clip_epsilon=_positive_float(
            algorithm_raw["clip_epsilon"], "algorithm.clip_epsilon"
        ),
        value_coefficient=_positive_float(
            algorithm_raw["value_coefficient"], "algorithm.value_coefficient"
        ),
        entropy_coefficient=_probability(
            algorithm_raw["entropy_coefficient"],
            "algorithm.entropy_coefficient",
        ),
        rollout_steps=_positive_int(
            algorithm_raw["rollout_steps"], "algorithm.rollout_steps"
        ),
        update_epochs=_positive_int(
            algorithm_raw["update_epochs"], "algorithm.update_epochs"
        ),
        minibatch_size=_positive_int(
            algorithm_raw["minibatch_size"], "algorithm.minibatch_size"
        ),
        max_gradient_norm=_positive_float(
            algorithm_raw["max_gradient_norm"], "algorithm.max_gradient_norm"
        ),
    )
    if algorithm.minibatch_size > algorithm.rollout_steps:
        raise ConfigurationError(
            "algorithm.minibatch_size cannot exceed algorithm.rollout_steps"
        )

    reward_raw = _section(raw["reward"], "reward")
    _check_keys(reward_raw, "reward", required={"kind", "terminal_penalty"})
    if reward_raw["kind"] not in {
        "sparse_lines",
        "native_score",
        "privileged_shaped",
    }:
        raise ConfigurationError("reward.kind is not supported")
    if isinstance(reward_raw["terminal_penalty"], bool) or not isinstance(
        reward_raw["terminal_penalty"], (int, float)
    ):
        raise ConfigurationError("reward.terminal_penalty must be numeric")
    reward = RewardConfig(
        kind=reward_raw["kind"],
        terminal_penalty=float(reward_raw["terminal_penalty"]),
    )

    training_raw = _section(raw["training"], "training")
    _check_keys(
        training_raw,
        "training",
        required={
            "environment_steps",
            "checkpoint_interval",
            "output_directory",
            "device",
        },
    )
    if training_raw["device"] not in {"cpu", "mps", "cuda", "auto"}:
        raise ConfigurationError("training.device must be cpu, mps, cuda, or auto")
    if (
        not isinstance(training_raw["output_directory"], str)
        or not training_raw["output_directory"]
    ):
        raise ConfigurationError("training.output_directory must be a path string")
    training = TrainingConfig(
        environment_steps=_positive_int(
            training_raw["environment_steps"], "training.environment_steps"
        ),
        checkpoint_interval=_positive_int(
            training_raw["checkpoint_interval"], "training.checkpoint_interval"
        ),
        output_directory=Path(training_raw["output_directory"]),
        device=training_raw["device"],
    )

    evaluation_raw = _section(raw["evaluation"], "evaluation")
    _check_keys(
        evaluation_raw,
        "evaluation",
        required={
            "episodes",
            "first_seed",
            "max_steps_per_episode",
            "output_directory",
        },
    )
    if isinstance(evaluation_raw["first_seed"], bool) or not isinstance(
        evaluation_raw["first_seed"], int
    ):
        raise ConfigurationError("evaluation.first_seed must be an integer")
    if (
        not isinstance(evaluation_raw["output_directory"], str)
        or not evaluation_raw["output_directory"]
    ):
        raise ConfigurationError("evaluation.output_directory must be a path string")
    evaluation = EvaluationConfig(
        episodes=_positive_int(evaluation_raw["episodes"], "evaluation.episodes"),
        first_seed=evaluation_raw["first_seed"],
        max_steps_per_episode=_positive_int(
            evaluation_raw["max_steps_per_episode"],
            "evaluation.max_steps_per_episode",
        ),
        output_directory=Path(evaluation_raw["output_directory"]),
    )

    return VisualExperimentConfig(
        name=raw["name"],
        seed=raw["seed"],
        observation=observation,
        actions=actions,
        model=model,
        algorithm=algorithm,
        reward=reward,
        training=training,
        evaluation=evaluation,
    )
