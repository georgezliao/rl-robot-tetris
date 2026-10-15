# RL Robot Tetris

An experimental Tetris environment for comparing agents that learn from the raw
board with agents that learn from engineered board features.

## Research questions

1. How does a CNN trained on a raw board grid compare with a feature-based model?
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
  observations/          Raw-grid and engineered-feature encoders
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
- `control`: choose left, right, rotate, soft drop, hard drop, or no-op each tick.

Two primary observation modes are planned:

- `raw_grid`: board channels suitable for a CNN.
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
synchronous subprocess bridge. Real-time gravity is disabled for RL episodes so
the board changes only when the agent acts, and the piece generator is seedable.

Build the bridge and run a smoke episode with the random agent:

```bash
./scripts/build_java_bridge.sh
PYTHONPATH=src python scripts/random_java_agent.py --seed 42
```

The bridge implements the controller action mode. The placement-level adapter
and learned agent are separate next milestones.

### Java bridge protocol

`JavaFxControlEnv` starts one Java process and sends `RESET <seed>` or
`STEP <action-id>` commands over standard input. Each command returns the raw
20×10 board, reward, terminal flag, and episode metrics as JSON. It implements:

```python
with JavaFxControlEnv() as env:
    observation = env.reset(seed=42)
    observation, reward, done, info = env.step(ControlAction.HARD_DROP)
```

On macOS the scripts automatically use the installed Liberica Java 8 JDK. Set
`JAVAFX_JAVA_HOME` when JavaFX is installed under a different JDK.

## Experimental discipline

- Use the same piece sequences and evaluation seeds across agents.
- Keep observation, action-mode, and reward-shaping ablations separate.
- Report lines, pieces, score, holes, and survival time—not reward alone.
- Evaluate on held-out seeds and record wall-clock time and environment steps.
- Treat external engines as separately configured reference baselines; do not
  claim direct comparability until rules, previews, hold, and timing match.

