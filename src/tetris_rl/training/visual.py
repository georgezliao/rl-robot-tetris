"""Training orchestration for a CNN policy controlling the JavaFX UI."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from tetris_rl.envs.javafx import JavaFxControlEnv
from tetris_rl.models.cnn_policy import VisualActorCritic
from tetris_rl.training.config import VisualExperimentConfig
from tetris_rl.training.ppo import (
    generalized_advantage_estimates,
    ppo_update,
)
from tetris_rl.types import ControlAction


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Locations and counters produced by one bounded training run."""

    run_directory: Path
    environment_steps: int
    completed_episodes: int
    updates: int


def action_set(config: VisualExperimentConfig) -> tuple[ControlAction, ...]:
    """Return policy-index actions in a stable order."""
    actions = tuple(ControlAction)
    if not config.actions.allow_hold:
        actions = tuple(
            action for action in actions if action is not ControlAction.HOLD
        )
    return actions


def validate_runtime_support(config: VisualExperimentConfig) -> None:
    """Reject valid experimental settings not yet implemented by this trainer."""
    unsupported: list[str] = []
    if config.observation.region != "full_ui":
        unsupported.append("only observation.region=full_ui is implemented")
    if config.observation.color != "rgb":
        unsupported.append("only observation.color=rgb is implemented")
    if config.actions.gravity_ticks_per_step != 1:
        unsupported.append("only one gravity tick per step is implemented")
    if config.actions.macro_actions:
        unsupported.append("macro actions are not implemented")
    if config.reward.kind != "sparse_lines":
        unsupported.append("only reward.kind=sparse_lines is implemented")
    if unsupported:
        raise ValueError("; ".join(unsupported))


def resolve_device(name: str) -> torch.device:
    """Resolve a requested accelerator and fail rather than silently changing it."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(name)


def experiment_summary(config: VisualExperimentConfig) -> dict[str, Any]:
    """Describe the resolved model and data contract without starting the game."""
    validate_runtime_support(config)
    actions = action_set(config)
    model = VisualActorCritic(
        config.observation.shape, len(actions), config.model.feature_dim
    )
    device = resolve_device(config.training.device)
    return {
        "name": config.name,
        "observation_shape": list(config.observation.shape),
        "actions": [action.name for action in actions],
        "device": str(device),
        "model_parameters": model.parameter_count,
        "environment_steps": config.training.environment_steps,
        "status": "validated_without_environment_execution",
    }


def _tensor_observation(observation: np.ndarray, device: torch.device) -> Tensor:
    return torch.from_numpy(np.ascontiguousarray(observation)).unsqueeze(0).to(device)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def _checkpoint(
    path: Path,
    model: VisualActorCritic,
    optimizer: torch.optim.Optimizer,
    environment_steps: int,
    config: VisualExperimentConfig,
) -> None:
    torch.save(
        {
            "schema_version": 1,
            "environment_steps": environment_steps,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config.to_dict(),
        },
        path,
    )


def train_visual_agent(
    config: VisualExperimentConfig,
    *,
    environment_steps_override: int | None = None,
) -> TrainingResult:
    """Train the reference PPO policy for a finite number of UI interactions."""
    validate_runtime_support(config)
    requested_steps = (
        config.training.environment_steps
        if environment_steps_override is None
        else environment_steps_override
    )
    if requested_steps < 1:
        raise ValueError("environment steps must be positive")

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = resolve_device(config.training.device)
    actions = action_set(config)
    model = VisualActorCritic(
        config.observation.shape, len(actions), config.model.feature_dim
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.algorithm.learning_rate)

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    run_directory = config.training.output_directory / f"{config.name}-{run_id}"
    checkpoint_directory = run_directory / "checkpoints"
    checkpoint_directory.mkdir(parents=True)
    _write_json(
        run_directory / "resolved_config.json",
        {
            **config.to_dict(),
            "executed_environment_steps": requested_steps,
            "action_labels": [action.name for action in actions],
            "resolved_device": str(device),
            "model_parameters": model.parameter_count,
        },
    )
    metrics_path = run_directory / "metrics.jsonl"

    steps_done = 0
    completed_episodes = 0
    updates = 0
    episode_index = 0
    episode_return = 0.0
    episode_length = 0
    next_checkpoint = config.training.checkpoint_interval

    with JavaFxControlEnv(
        observation_size=(config.observation.height, config.observation.width),
        frame_stack=config.observation.frame_stack,
    ) as environment:
        observation = environment.reset(seed=config.seed)
        while steps_done < requested_steps:
            rollout_length = min(
                config.algorithm.rollout_steps, requested_steps - steps_done
            )
            observations: list[Tensor] = []
            selected_actions: list[Tensor] = []
            log_probabilities: list[Tensor] = []
            values: list[Tensor] = []
            rewards: list[float] = []
            dones: list[bool] = []

            model.eval()
            for _ in range(rollout_length):
                observation_tensor = _tensor_observation(observation, device)
                policy_action, log_probability, value = model.act(observation_tensor)
                action_index = int(policy_action.item())
                next_observation, raw_reward, done, info = environment.step(
                    actions[action_index]
                )
                reward = raw_reward + (config.reward.terminal_penalty if done else 0.0)

                observations.append(observation_tensor.squeeze(0))
                selected_actions.append(policy_action.squeeze(0))
                log_probabilities.append(log_probability.squeeze(0))
                values.append(value.squeeze(0))
                rewards.append(reward)
                dones.append(done)
                steps_done += 1
                episode_length += 1
                episode_return += reward

                if done:
                    _append_jsonl(
                        metrics_path,
                        {
                            "type": "episode",
                            "episode": episode_index,
                            "seed": config.seed + episode_index,
                            "environment_steps": steps_done,
                            "length": episode_length,
                            "return": episode_return,
                            "score": info.get("score"),
                            "total_lines": info.get("total_lines"),
                            "pieces_placed": info.get("pieces_placed"),
                        },
                    )
                    completed_episodes += 1
                    episode_index += 1
                    episode_return = 0.0
                    episode_length = 0
                    next_observation = environment.reset(
                        seed=config.seed + episode_index
                    )
                observation = next_observation

            with torch.no_grad():
                if dones[-1]:
                    next_value = torch.zeros((), device=device)
                else:
                    _, next_values = model(_tensor_observation(observation, device))
                    next_value = next_values.squeeze(0)

            observation_batch = torch.stack(observations)
            action_batch = torch.stack(selected_actions).long()
            old_log_probability_batch = torch.stack(log_probabilities)
            value_batch = torch.stack(values)
            reward_batch = torch.tensor(rewards, dtype=torch.float32, device=device)
            done_batch = torch.tensor(dones, dtype=torch.float32, device=device)
            advantages, returns = generalized_advantage_estimates(
                reward_batch,
                value_batch,
                done_batch,
                next_value,
                gamma=config.algorithm.gamma,
                gae_lambda=config.algorithm.gae_lambda,
            )
            update_metrics = ppo_update(
                model,
                optimizer,
                observation_batch,
                action_batch,
                old_log_probability_batch,
                advantages,
                returns,
                config.algorithm,
            )
            updates += 1
            _append_jsonl(
                metrics_path,
                {
                    "type": "update",
                    "update": updates,
                    "environment_steps": steps_done,
                    **{
                        field: getattr(update_metrics, field)
                        for field in update_metrics.__dataclass_fields__
                    },
                },
            )

            if steps_done >= next_checkpoint:
                _checkpoint(
                    checkpoint_directory / f"step-{steps_done}.pt",
                    model,
                    optimizer,
                    steps_done,
                    config,
                )
                while next_checkpoint <= steps_done:
                    next_checkpoint += config.training.checkpoint_interval

    _checkpoint(checkpoint_directory / "final.pt", model, optimizer, steps_done, config)
    _write_json(
        run_directory / "summary.json",
        {
            "environment_steps": steps_done,
            "completed_episodes": completed_episodes,
            "updates": updates,
            "unfinished_episode_length": episode_length,
            "unfinished_episode_return": episode_return,
        },
    )
    return TrainingResult(run_directory, steps_done, completed_episodes, updates)
