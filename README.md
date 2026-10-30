# RL Robot Tetris

This project is attending as an experimental environment on one of my favorite games, Tetris. I want to compare agents that learn by watching
and controlling the rendered game UI with agents that receive engineered state, mainly to see what conditions agents learn fastest under. In addition, this will be a chance for me to compare different architectures, reward definitions, and encoders to understand what is effective.

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
tetris/                  Original JavaFX game and RlBridge.java (nested repo)
configs/                 Experiment configuration (visual/ is the live one)
docs/                    Architecture notes
experiments/             Run outputs (ignored by Git)
notebooks/               Exploration and result analysis
scripts/                 Bridge, recording, replay, and stability commands
src/tetris_rl/
  envs/javafx.py         JavaFxControlEnv: the environment the agent uses
  models/                CNN encoder and actor-critic policy
  training/              Config, PPO, training loop, and checkpoint evaluation
  events/                Episode manifest and step-record format
  core/, envs/control.py, envs/placement.py, observations/, rewards/,
  agents/, search/       Placeholders for the planned comparison conditions
tests/
  unit/                  Core board/features, config, policy, PPO, evaluation, records
  integration/           Environment contract tests
```

[docs/architecture.md](docs/architecture.md) follows one decision from the
JavaFX bridge to a saved checkpoint and lists which modules are placeholders.

## Environment contract

The project exposes the requested four-value API:

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
pip install -e '.[rl,dev]'
pytest
```

The Java sources have a Maven build pinned by the project wrapper. The build
uses JavaFX 25.0.1 and compiles for Java 25:

```bash
./mvnw clean compile
./mvnw javafx:run
```

Importing `pom.xml` into an editor supplies the JavaFX dependencies without a
machine-specific module path. The older bridge scripts retain their Liberica
Java 8 fallback because that path has also been exercised on the current macOS
development machine.

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
adapter remains a separate milestone.

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

## Reference visual agent

`configs/visual/reference_ppo.yaml` defines the first trainable visual-control
condition. The policy receives four consecutive RGB frames, concatenated into a
`12×128×96` tensor. A three-layer convolutional encoder feeds a categorical
policy head and a scalar value head. The policy chooses one of the seven UI
controls; it does not receive board cells, legal-action masks, score, line
counts, or other values from the bridge's `info` mapping.

The current reference optimizer is proximal policy optimization (PPO). For a
transition at index `t`, let `r_t` be the scalar reward, `V_t` the value
estimate, `d_t` in `{0, 1}` the terminal indicator, `gamma` the discount, and
`lambda` the generalized-advantage parameter. The implementation computes

```text
delta_t = r_t + gamma (1 - d_t) V_(t+1) - V_t
A_t = delta_t + gamma lambda (1 - d_t) A_(t+1)
R_t = A_t + V_t
```

The factor `1 - d_t` prevents either recursion or value bootstrapping across an
episode boundary. PPO then optimizes the clipped probability-ratio surrogate,
mean-squared value error, and an entropy bonus. The numerical coefficients and
rollout sizes are recorded in the resolved run configuration.

Validate the configuration, tensor contract, available device, action order,
and model construction without starting JavaFX:

```bash
PYTHONPATH=src python -m tetris_rl.training.train \
  --config configs/visual/reference_ppo.yaml --dry-run
```

Start training with the configured step budget, or use a small override to test
the complete data path:

```bash
PYTHONPATH=src python -m tetris_rl.training.train \
  --config configs/visual/reference_ppo.yaml

PYTHONPATH=src python -m tetris_rl.training.train \
  --config configs/visual/reference_ppo.yaml --environment-steps 8
```

Each run writes its resolved configuration, update and episode metrics, summary,
and checkpoints under `experiments/training/`. That directory is ignored by Git.
The step override is intended for bounded diagnostics; an eight-step run does
not provide evidence of policy improvement. The planned visual-agent ablation
configurations remain to be implemented.

### Checkpoint evaluation

The evaluation command reconstructs the CNN and action order, verifies that the
checkpoint observation, action, model, and reward conditions match the requested
configuration, and chooses actions deterministically. Evaluation seeds and the
per-episode step limit are declared under `evaluation` in the configuration.

```bash
PYTHONPATH=src python -m tetris_rl.training.evaluate \
  --config configs/visual/reference_ppo.yaml \
  --checkpoint experiments/training/<run>/checkpoints/final.pt
```

Use explicit overrides only for bounded diagnostic runs:

```bash
PYTHONPATH=src python -m tetris_rl.training.evaluate \
  --config configs/visual/reference_ppo.yaml \
  --checkpoint experiments/training/<run>/checkpoints/final.pt \
  --episodes 1 --max-steps-per-episode 8
```

Each evaluation writes generic episode manifests and JSONL step records,
per-episode results, `episodes.json`, and `summary.json` under
`experiments/evaluation/`. Aggregate output includes lines, score, pieces,
survival in pieces, and lines per 100 pieces. An episode that reaches the step
limit is marked as truncated rather than terminal. Lines per 100 pieces is stored
as `null` when no pieces were placed. The evaluator hashes model parameters
before and after execution and fails if they change.

## Episode records and replay

The recording format is game-neutral. Each episode directory contains:

- `manifest.json`: schema version, game and environment identifiers, seed,
  action labels, observation specification, configuration, and model identifier.
- `steps.jsonl`: one `StepEvent` per line, ordered from step zero.

A step records its action, scalar reward, cumulative reward, terminal flag,
game-specific scalar metrics, and a SHA-256 digest of the numerical observation.
Frame bytes are not embedded in JSONL; future visualizer frames should be stored
as separate artifacts and referenced by path.

Record a random-policy episode:

```bash
PYTHONPATH=src python scripts/record_tetris_episode.py \
  --output-root experiments/records --seed 42 --action-seed 42
```

Replay the recorded seed and action sequence:

```bash
PYTHONPATH=src python scripts/replay_tetris_episode.py \
  experiments/records/<episode-id>
```

By default, replay compares observation digests, rewards, termination, and all
recorded metrics at every step. `--ignore-pixels` disables only the digest check;
it is intended for comparisons across rendering platforms where font or raster
differences may occur.

## Stability checks

The bounded stability command checks the `12×128×96` `float32` observation
contract, finite values in `[0, 1]`, same-process replay for an identical seed
and action sequence, repeated reset/step cycles, and bridge shutdown:

```bash
PYTHONPATH=src python scripts/stability_check.py \
  --seed 42 --determinism-steps 100 --episodes 10 --max-steps 500
```

This command does not measure resident-memory growth, establish pixel equality
across operating systems, or provide evidence of learning convergence. Its JSON
report lists these exclusions explicitly.

## Experimental discipline

- Use the same piece sequences and evaluation seeds across agents.
- Keep observation, action-mode, and reward-shaping ablations separate.
- Report lines, pieces, score, holes, and survival time and not just reward alone.
- Evaluate on held-out seeds and record wall-clock time and environment steps.
- Treat external engines as separately configured reference baselines; do not
  claim direct comparability until rules, previews, hold, and timing match.

