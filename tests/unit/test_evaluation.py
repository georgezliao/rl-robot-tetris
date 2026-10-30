from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from tetris_rl.events.recorder import read_episode
from tetris_rl.models.cnn_policy import VisualActorCritic
from tetris_rl.training.config import VisualExperimentConfig, load_visual_config
from tetris_rl.training.evaluate import (
    CheckpointCompatibilityError,
    LoadedCheckpoint,
    evaluate_loaded_policy,
    load_visual_checkpoint,
)
from tetris_rl.training.visual import action_set
from tetris_rl.types import ControlAction

REFERENCE_CONFIG = (
    Path(__file__).resolve().parents[2] / "configs" / "visual" / "reference_ppo.yaml"
)


def reference_config() -> VisualExperimentConfig:
    return load_visual_config(REFERENCE_CONFIG)


def save_checkpoint(
    path: Path,
    config: VisualExperimentConfig,
    model: VisualActorCritic,
) -> None:
    torch.save(
        {
            "schema_version": 1,
            "environment_steps": 123,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "config": config.to_dict(),
        },
        path,
    )


class TwoStepEnvironment:
    def __init__(self, shape: tuple[int, int, int], *, terminates: bool = True) -> None:
        self.shape = shape
        self.terminates = terminates
        self.seeds: list[int] = []
        self.steps = 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        assert seed is not None
        self.seeds.append(seed)
        self.steps = 0
        return np.zeros(self.shape, dtype=np.float32)

    def step(
        self, action: ControlAction | int
    ) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        assert int(action) in {int(item) for item in ControlAction}
        self.steps += 1
        done = self.terminates and self.steps == 2
        lines = 1 if done else 0
        observation = np.full(self.shape, self.steps / 10, dtype=np.float32)
        return (
            observation,
            float(lines),
            done,
            {
                "lines_cleared": lines,
                "total_lines": lines,
                "score": 100 * lines,
                "pieces_placed": lines,
            },
        )


def loaded_policy(config: VisualExperimentConfig) -> LoadedCheckpoint:
    actions = action_set(config)
    model = VisualActorCritic(
        config.observation.shape,
        len(actions),
        config.model.feature_dim,
    )
    return LoadedCheckpoint(
        model=model,
        sha256="a" * 64,
        environment_steps=123,
        action_labels=tuple(action.name.lower() for action in actions),
    )


def test_checkpoint_reconstructs_model_and_action_order(tmp_path: Path) -> None:
    config = reference_config()
    actions = action_set(config)
    model = VisualActorCritic(
        config.observation.shape,
        len(actions),
        config.model.feature_dim,
    )
    checkpoint = tmp_path / "final.pt"
    save_checkpoint(checkpoint, config, model)

    loaded = load_visual_checkpoint(checkpoint, config)

    assert loaded.environment_steps == 123
    assert loaded.action_labels == tuple(action.name.lower() for action in actions)
    for expected, actual in zip(model.parameters(), loaded.model.parameters()):
        torch.testing.assert_close(actual, expected)


def test_checkpoint_rejects_incompatible_model_configuration(
    tmp_path: Path,
) -> None:
    config = reference_config()
    model = VisualActorCritic(
        config.observation.shape,
        len(action_set(config)),
        config.model.feature_dim,
    )
    checkpoint = tmp_path / "final.pt"
    save_checkpoint(checkpoint, config, model)
    incompatible = replace(config, model=replace(config.model, feature_dim=256))

    with pytest.raises(CheckpointCompatibilityError, match="model does not match"):
        load_visual_checkpoint(checkpoint, incompatible)


def test_evaluation_records_held_out_seeds_and_preserves_parameters(
    tmp_path: Path,
) -> None:
    config = reference_config()
    loaded = loaded_policy(config)
    before = {
        name: value.detach().clone()
        for name, value in loaded.model.state_dict().items()
    }
    environment = TwoStepEnvironment(config.observation.shape)

    result = evaluate_loaded_policy(
        config,
        loaded,
        environment,
        output_root=tmp_path,
        episode_count=2,
        max_steps_per_episode=5,
    )

    assert environment.seeds == [10000, 10001]
    assert [episode.steps for episode in result.episodes] == [2, 2]
    assert all(
        episode.terminated and not episode.truncated for episode in result.episodes
    )
    assert result.summary["total_lines"] == 2
    assert result.summary["total_score"] == 200
    assert result.summary["total_pieces"] == 2
    assert result.summary["lines_per_100_pieces"] == 100.0
    assert result.summary["parameters_unchanged"] is True
    for name, value in loaded.model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    for episode in result.episodes:
        recording = read_episode(episode.recording_directory)
        assert len(recording.steps) == 2
        assert recording.steps[-1].done is True
        assert recording.manifest.seed == episode.seed


def test_step_limit_marks_an_episode_truncated(tmp_path: Path) -> None:
    config = reference_config()
    environment = TwoStepEnvironment(config.observation.shape, terminates=False)

    result = evaluate_loaded_policy(
        config,
        loaded_policy(config),
        environment,
        output_root=tmp_path,
        episode_count=1,
        max_steps_per_episode=1,
    )

    assert result.episodes[0].terminated is False
    assert result.episodes[0].truncated is True
    assert result.summary["truncated_episodes"] == 1
    assert result.summary["lines_per_100_pieces"] is None
