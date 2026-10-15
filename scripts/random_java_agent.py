"""Run a random agent against the original JavaFX game bridge."""

import argparse

from tetris_rl.agents.random_agent import RandomAgent
from tetris_rl.envs.javafx import JavaFxControlEnv
from tetris_rl.types import ControlAction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=10_000)
    args = parser.parse_args()

    agent = RandomAgent(list(ControlAction), seed=args.seed)
    with JavaFxControlEnv() as env:
        observation = env.reset(seed=args.seed)
        total_reward = 0.0

        for step in range(1, args.max_steps + 1):
            action = agent.act(observation)
            observation, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break

    print(
        f"steps={step} reward={total_reward:.0f} "
        f"lines={info['total_lines']} score={info['score']} "
        f"pieces={info['pieces_placed']} done={done}"
    )


if __name__ == "__main__":
    main()
