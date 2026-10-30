import numpy as np
import torch

from tetris_rl.models.cnn_policy import VisualActorCritic
from tetris_rl.training.visual import EpisodeProgress, bootstrap_value, collect_rollout
from tetris_rl.types import ControlAction, Info, Observation, StepResult

SHAPE = (3, 36, 36)


class ThreeStepEnv:
    """Every episode ends on its third step; reward 1.0 per step."""

    def __init__(self) -> None:
        self.reset_seeds: list[int | None] = []
        self.steps_in_episode = 0

    def _observation(self) -> Observation:
        return np.full(SHAPE, self.steps_in_episode / 10, dtype=np.float32)

    def reset(self, seed: int | None = None) -> Observation:
        self.reset_seeds.append(seed)
        self.steps_in_episode = 0
        return self._observation()

    def step(self, action: ControlAction | int) -> StepResult:
        del action
        self.steps_in_episode += 1
        done = self.steps_in_episode == 3
        info: Info = {"score": 7, "total_lines": 0, "pieces_placed": 2}
        return self._observation(), 1.0, done, info


def test_rollout_crosses_an_episode_boundary_and_resets_on_the_next_seed() -> None:
    torch.manual_seed(0)
    environment = ThreeStepEnv()
    model = VisualActorCritic(SHAPE, action_count=7, feature_dim=8)
    progress = EpisodeProgress()
    observation = environment.reset(seed=42)

    rollout = collect_rollout(
        environment,
        model,
        tuple(ControlAction),
        observation,
        progress,
        length=5,
        first_seed=42,
        terminal_penalty=-2.0,
        steps_before=100,
        device=torch.device("cpu"),
    )

    assert len(rollout) == 5
    assert rollout.observations.shape == (5, *SHAPE)
    torch.testing.assert_close(rollout.dones, torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]))
    # The terminal penalty is added only on the step that ends the episode.
    torch.testing.assert_close(
        rollout.rewards, torch.tensor([1.0, 1.0, -1.0, 1.0, 1.0])
    )
    # Row 3 is the first observation of the new episode, not the terminal frame.
    assert rollout.observations[3].max().item() == 0.0
    assert environment.reset_seeds == [42, 43]
    assert rollout.finished_episodes == (
        {
            "type": "episode",
            "episode": 0,
            "seed": 42,
            "environment_steps": 103,
            "length": 3,
            "return": 1.0,
            "score": 7,
            "total_lines": 0,
            "pieces_placed": 2,
        },
    )
    assert (progress.index, progress.length, progress.total_return) == (1, 2, 2.0)


def test_bootstrap_value_is_zero_after_a_terminal_step() -> None:
    torch.manual_seed(0)
    environment = ThreeStepEnv()
    model = VisualActorCritic(SHAPE, action_count=7, feature_dim=8)
    rollout = collect_rollout(
        environment,
        model,
        tuple(ControlAction),
        environment.reset(seed=0),
        EpisodeProgress(),
        length=3,
        first_seed=0,
        terminal_penalty=0.0,
        steps_before=0,
        device=torch.device("cpu"),
    )

    assert rollout.dones[-1].item() == 1.0
    assert bootstrap_value(model, rollout, torch.device("cpu")).item() == 0.0
