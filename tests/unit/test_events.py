import json

import pytest

from tetris_rl.events.recorder import EpisodeRecorder, read_episode
from tetris_rl.events.schema import EpisodeManifest, StepEvent


def manifest() -> EpisodeManifest:
    return EpisodeManifest.create(
        episode_id="episode-test",
        game_id="test-game",
        environment_version="test-v1",
        seed=7,
        action_spec=("wait", "move"),
        observation_spec={"shape": "3x8x8", "dtype": "float32"},
    )


def test_episode_round_trip(tmp_path) -> None:
    episode_manifest = manifest()
    with EpisodeRecorder(tmp_path, episode_manifest) as recorder:
        recorder.record(
            StepEvent(
                episode_id=episode_manifest.episode_id,
                step_index=0,
                action=1,
                action_label="move",
                reward=0.5,
                cumulative_reward=0.5,
                done=False,
                metrics={"score": 1},
                observation_digest="abc",
            )
        )

    loaded = read_episode(tmp_path)
    assert loaded.manifest == episode_manifest
    assert len(loaded.steps) == 1
    assert loaded.steps[0].metrics == {"score": 1}
    assert json.loads((tmp_path / "steps.jsonl").read_text())["action"] == 1


def test_recorder_rejects_nonconsecutive_steps(tmp_path) -> None:
    episode_manifest = manifest()
    with (
        EpisodeRecorder(tmp_path, episode_manifest) as recorder,
        pytest.raises(ValueError, match="step_index"),
    ):
        recorder.record(
            StepEvent(
                episode_id=episode_manifest.episode_id,
                step_index=1,
                action=0,
                action_label="wait",
                reward=0.0,
                cumulative_reward=0.0,
                done=False,
            )
        )


def test_recorder_rejects_inconsistent_cumulative_reward(tmp_path) -> None:
    episode_manifest = manifest()
    with (
        EpisodeRecorder(tmp_path, episode_manifest) as recorder,
        pytest.raises(ValueError, match="cumulative_reward"),
    ):
        recorder.record(
            StepEvent(
                episode_id=episode_manifest.episode_id,
                step_index=0,
                action=0,
                action_label="wait",
                reward=1.0,
                cumulative_reward=2.0,
                done=False,
            )
        )
