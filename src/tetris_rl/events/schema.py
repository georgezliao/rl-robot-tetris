"""Versioned, game-neutral schemas for recorded reinforcement-learning episodes."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, TypeAlias
from uuid import uuid4

JsonScalar: TypeAlias = str | int | float | bool | None
SCHEMA_VERSION = 1


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, received {value!r}")


def _validate_metrics(metrics: dict[str, JsonScalar]) -> None:
    for key, value in metrics.items():
        if not key:
            raise ValueError("metric names must not be empty")
        if isinstance(value, float):
            _require_finite(f"metric {key!r}", value)


@dataclass(frozen=True, slots=True)
class EpisodeManifest:
    """Metadata needed to interpret or reproduce one recorded episode.

    ``started_at`` uses an ISO 8601 UTC timestamp. ``action_spec[i]`` names
    action integer ``i``. Configuration and observation specifications must be
    JSON-serializable, but their keys remain game-defined.
    """

    episode_id: str
    game_id: str
    environment_version: str
    seed: int
    started_at: str
    action_spec: tuple[str, ...]
    observation_spec: dict[str, JsonScalar]
    configuration: dict[str, Any] = field(default_factory=dict)
    model_id: str | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("episode_id", "game_id", "environment_version", "started_at"):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema version {self.schema_version}; "
                f"expected {SCHEMA_VERSION}"
            )
        if not self.action_spec or any(not label for label in self.action_spec):
            raise ValueError("action_spec must contain non-empty action labels")
        try:
            parsed_started_at = datetime.fromisoformat(self.started_at)
        except ValueError as exception:
            raise ValueError("started_at must be an ISO 8601 timestamp") from exception
        if parsed_started_at.tzinfo is None:
            raise ValueError("started_at must include a UTC offset")
        if parsed_started_at.utcoffset() != timedelta(0):
            raise ValueError("started_at must use UTC")

    @classmethod
    def create(
        cls,
        *,
        game_id: str,
        environment_version: str,
        seed: int,
        action_spec: tuple[str, ...],
        observation_spec: dict[str, JsonScalar],
        configuration: dict[str, Any] | None = None,
        model_id: str | None = None,
        episode_id: str | None = None,
    ) -> EpisodeManifest:
        return cls(
            episode_id=episode_id or str(uuid4()),
            game_id=game_id,
            environment_version=environment_version,
            seed=seed,
            started_at=datetime.now(UTC).isoformat(),
            action_spec=action_spec,
            observation_spec=observation_spec,
            configuration=dict(configuration or {}),
            model_id=model_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EpisodeManifest:
        data = dict(value)
        data["action_spec"] = tuple(data["action_spec"])
        return cls(**data)


@dataclass(frozen=True, slots=True)
class StepEvent:
    """One observed transition, indexed from zero within an episode.

    Rewards and cumulative rewards are finite scalar values. ``metrics`` may
    contain game-specific aggregate values. A frame reference points to an
    external image or video artifact; image bytes are not embedded in JSONL.
    """

    episode_id: str
    step_index: int
    action: int
    action_label: str
    reward: float
    cumulative_reward: float
    done: bool
    metrics: dict[str, JsonScalar] = field(default_factory=dict)
    observation_digest: str | None = None
    frame_reference: str | None = None
    model_output: dict[str, Any] | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id must not be empty")
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")
        if self.action < 0:
            raise ValueError("action must be non-negative")
        if not self.action_label:
            raise ValueError("action_label must not be empty")
        _require_finite("reward", self.reward)
        _require_finite("cumulative_reward", self.cumulative_reward)
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema version {self.schema_version}; "
                f"expected {SCHEMA_VERSION}"
            )
        _validate_metrics(self.metrics)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> StepEvent:
        return cls(**value)
