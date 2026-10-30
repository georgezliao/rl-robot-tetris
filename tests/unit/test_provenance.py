import json
import subprocess
from pathlib import Path
from typing import Self

import numpy as np
import pytest
import torch
import yaml

from tetris_rl.training import visual
from tetris_rl.training.config import load_visual_config
from tetris_rl.training.provenance import collect_provenance, git_state

REFERENCE_CONFIG = Path("configs/visual/reference_ppo.yaml")


def _git(directory: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=directory,
        check=True,
        capture_output=True,
    )


def test_git_state_reports_commit_and_dirty_flag(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    (tmp_path / "a.txt").write_text("one", encoding="utf-8")
    _git(tmp_path, "add", "a.txt")
    _git(tmp_path, "commit", "-q", "-m", "first")

    clean = git_state(tmp_path)
    (tmp_path / "a.txt").write_text("two", encoding="utf-8")
    dirty = git_state(tmp_path)

    assert len(clean["commit"]) == 40
    assert clean["dirty"] is False
    assert dirty == {"commit": clean["commit"], "dirty": True}


def test_git_state_is_empty_outside_a_repository(tmp_path: Path) -> None:
    assert git_state(tmp_path) == {"commit": None, "dirty": None}


def test_collect_provenance_fields(tmp_path: Path) -> None:
    record = collect_provenance(
        config_path=Path("configs/x.yaml"),
        seeds={"training_first_seed": 42},
        device=torch.device("cpu"),
        extra={"checkpoint_sha256": "abc"},
        repository_root=tmp_path,
    )

    assert set(record) == {
        "schema_version",
        "created_utc",
        "repository",
        "tetris",
        "config_path",
        "seeds",
        "device",
        "platform",
        "versions",
        "extra",
    }
    assert record["config_path"] == "configs/x.yaml"
    assert record["seeds"] == {"training_first_seed": 42}
    assert record["device"]["type"] == "cpu"
    assert record["device"]["description"]
    assert record["versions"]["torch"] == torch.__version__
    # tmp_path has no scripts/java_home.sh and is not a Git checkout.
    assert record["versions"]["java"] is None
    assert record["repository"] == {"commit": None, "dirty": None}
    assert record["extra"] == {"checkpoint_sha256": "abc"}


class FakeJavaFxEnv:
    """Stands in for JavaFxControlEnv: blank frames, every episode lasts 3 steps."""

    def __init__(self, *, observation_size: tuple[int, int], frame_stack: int) -> None:
        self.shape = (3 * frame_stack, *observation_size)
        self.steps = 0

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def reset(self, seed: int | None = None) -> np.ndarray:
        self.steps = 0
        return np.zeros(self.shape, dtype=np.float32)

    def step(self, action: object) -> tuple[np.ndarray, float, bool, dict]:
        self.steps += 1
        return np.zeros(self.shape, dtype=np.float32), 0.0, self.steps == 3, {}


def test_training_run_writes_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = yaml.safe_load(REFERENCE_CONFIG.read_text(encoding="utf-8"))
    raw["observation"].update(height=36, width=36, frame_stack=1)
    raw["model"]["feature_dim"] = 8
    raw["algorithm"].update(rollout_steps=4, minibatch_size=2, update_epochs=1)
    raw["training"].update(
        environment_steps=4, checkpoint_interval=4, output_directory=str(tmp_path)
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    monkeypatch.setattr(visual, "JavaFxControlEnv", FakeJavaFxEnv)

    result = visual.train_visual_agent(
        load_visual_config(config_path), config_path=config_path
    )

    record = json.loads((result.run_directory / "provenance.json").read_text())
    assert record["config_path"] == str(config_path)
    assert record["seeds"] == {
        "training_first_seed": 42,
        "evaluation_seeds": [10000, 10099],
    }
