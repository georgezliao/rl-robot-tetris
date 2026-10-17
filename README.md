# RL Robot Tetris

An experimental Tetris environment for comparing agents that learn by watching
and controlling the rendered game UI with agents that receive engineered state.

## Research questions

1. How does a CNN trained on rendered UI pixels compare with a feature-based model?
2. How much does reward shaping (lines, holes, height, and bumpiness) help?
3. How does choosing a final placement compare with issuing one controller input
   at a time?
4. How do learned policies compare with search baselines such as beam search and
   Monte Carlo tree search?

The game engine is shared by every experiment so the observation, action, and
reward choices can be varied independently.

## Layout

```text
configs/                 Experiment configuration
experiments/             Saved run metadata (large outputs are ignored)
notebooks/               Exploration and result analysis
scripts/                 Command-line entry points
src/tetris_rl/
  core/                  Board, pieces, randomizer, and game rules
  envs/                  RL API and the two action-mode environments
  observations/          Rendered-pixel, raw-grid, and feature encoders
  rewards/               Sparse and shaped reward definitions
  agents/                Random, deep-RL, and heuristic policies
  search/                Beam-search and MCTS baselines
  training/              Training, evaluation, and metrics orchestration
tests/
  unit/                  Rules, encoders, and reward tests
  integration/           Environment contract and reproducibility tests
```

## Environment contract

The project deliberately exposes the requested four-value API:

```python
observation = env.reset(seed=42)
observation, reward, done, info = env.step(action)
```

Two action modes are planned:

- `placement`: choose rotation and column; the environment hard-drops the piece.
- `control`: press left, right, rotate, soft drop, hard drop, hold, or no-op.

The primary observation modes are:

- `rendered_rgb`: stacked screenshots of the complete JavaFX UI for a CNN. No
  board cells, coordinates, piece IDs, or legal-action masks are exposed.
- `raw_grid`: privileged board channels used only as a comparison condition.
- `features`: lines cleared, holes, aggregate/max height, bumpiness, wells, and
  related landing features. These measurements are inputs, not a hand-authored
  rule for how the agent must combine them.

## Getting started

Requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

The original JavaFX game in `tetris/` is connected to Python through a small,
synchronous subprocess bridge. The agent sees resized RGB screenshots of the
actual UI and acts through the same JavaFX key-event handler as a person. One
deterministic gravity tick follows each action, and the piece generator is seeded.

Build the bridge and run a smoke episode with the random agent:

```bash
./scripts/build_java_bridge.sh
PYTHONPATH=src python scripts/random_java_agent.py --seed 42
```

The bridge implements the visual controller action mode. The placement-level
adapter and learned CNN agent are separate next milestones.

### Java bridge protocol

`JavaFxControlEnv` starts one Java process and sends `RESET <seed>` or
`STEP <action-id>` commands over standard input. Each command returns a rendered
RGB frame, reward, terminal flag, and episode metrics as JSON. Python downsamples
and stacks four frames into a `12×128×96` CNN observation. It implements:

```python
with JavaFxControlEnv() as env:
    observation = env.reset(seed=42)
    observation, reward, done, info = env.step(ControlAction.HARD_DROP)
```

Reward and termination use internal game counters as privileged training signals,
but those counters are not added to the agent's pixel observation. The available
actions exactly match the existing UI: no-op, left, right, rotate, down, space,
and `C` for hold.

On macOS the scripts automatically use the installed Liberica Java 8 JDK. Set
`JAVAFX_JAVA_HOME` when JavaFX is installed under a different JDK.

## Experimental discipline

- Use the same piece sequences and evaluation seeds across agents.
- Keep observation, action-mode, and reward-shaping ablations separate.
- Report lines, pieces, score, holes, and survival time and not just reward alone.
- Evaluate on held-out seeds and record wall-clock time and environment steps.
- Treat external engines as separately configured reference baselines; do not
  claim direct comparability until rules, previews, hold, and timing match.

