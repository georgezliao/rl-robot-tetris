from pathlib import Path

import pytest

from tetris_rl.training.config import ConfigurationError, load_visual_config
from tetris_rl.training.visual import action_set, experiment_summary
from tetris_rl.types import ControlAction

REFERENCE_CONFIG = (
    Path(__file__).resolve().parents[2] / "configs" / "visual" / "reference_ppo.yaml"
)


def test_reference_visual_configuration_resolves_model_contract() -> None:
    config = load_visual_config(REFERENCE_CONFIG)

    assert config.observation.shape == (12, 128, 96)
    assert config.evaluation.first_seed == 10000
    assert config.evaluation.max_steps_per_episode == 10000
    assert action_set(config) == tuple(ControlAction)
    summary = experiment_summary(config)
    assert summary["status"] == "validated_without_environment_execution"
    assert summary["model_parameters"] > 0


def test_unknown_configuration_key_is_rejected(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(
        REFERENCE_CONFIG.read_text(encoding="utf-8") + "\nunknown: true\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="unknown keys"):
        load_visual_config(invalid)


def test_relative_configuration_inheritance_merges_nested_values(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent.yaml"
    child = tmp_path / "child.yaml"
    parent.write_text(REFERENCE_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    child.write_text(
        "extends: parent.yaml\nname: no_hold\nactions:\n  allow_hold: false\n",
        encoding="utf-8",
    )

    config = load_visual_config(child)

    assert config.name == "no_hold"
    assert ControlAction.HOLD not in action_set(config)
    assert len(action_set(config)) == 6


def test_terminal_variant_changes_only_penalty_name_and_evaluation_seeds() -> None:
    base = load_visual_config("configs/visual/reference_ppo.yaml").to_dict()
    variant = load_visual_config("configs/visual/reference_ppo_terminal.yaml").to_dict()

    assert variant["reward"]["terminal_penalty"] == -1.0
    assert variant["evaluation"]["first_seed"] == 1_000_000
    for changed in (
        ("name",),
        ("reward", "terminal_penalty"),
        ("evaluation", "first_seed"),
    ):
        parent_base, parent_variant = base, variant
        for key in changed[:-1]:
            parent_base, parent_variant = parent_base[key], parent_variant[key]
        parent_base.pop(changed[-1])
        parent_variant.pop(changed[-1])
    assert variant == base
