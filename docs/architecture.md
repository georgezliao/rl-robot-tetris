# Architecture

This page follows one agent decision through the code, from the JavaFX game to
a saved checkpoint, and lists which modules are placeholders. For the training
math and configuration keys, see the README's "Reference visual agent" section.

## The live path

```text
tetris/RlBridge.java          JavaFX game + line-oriented stdin/stdout protocol
        |  JSON: RGB frame, reward, done, info
        v
envs/javafx.py                JavaFxControlEnv: decode, resize, stack 4 frames
        |  float32 observation, shape (12, 128, 96)
        v
models/cnn_policy.py          VisualActorCritic: CNN encoder -> policy + value
        |  action index -> ControlAction
        v
training/visual.py            rollout collection, GAE, PPO update, checkpoints
training/ppo.py               generalized_advantage_estimates, ppo_update
        |  experiments/training/<run>/checkpoints/*.pt
        v
training/evaluate.py          deterministic policy on held-out seeds
                              -> experiments/evaluation/<run>/summary.json
```

Everything else in `src/tetris_rl/` is either support code for this path
(`training/config.py`, `events/`) or a placeholder (listed below).

### 1. The game and the bridge (`tetris/`)

`tetris/` holds the original JavaFX game. `RlBridge.java` launches it and reads
one command per line on stdin:

| Command | Effect | Response |
|---|---|---|
| `RESET <seed>` | Seeds `PieceFactory`, builds a fresh game and scene | Frame, reward `0.0` |
| `STEP <action-id>` | Fires one `KEY_PRESSED` event at the game pane, then advances gravity once unless that key already locked a piece | Frame, reward = lines cleared by this step |
| `CLOSE` | Exits the JavaFX platform | none |

Each response is one JSON line holding the RGB snapshot of the whole UI
(base64, `rgb24-base64`), `reward`, `done`, and an `info` object with
`lines_cleared`, `total_lines`, `score`, and `pieces_placed`. Commands run on
the JavaFX thread and the reader waits for each one to finish, so the game never
advances between agent decisions. Diagnostics go to stderr so stdout stays
machine-readable.

`scripts/run_java_bridge.sh` builds the Java sources into `build/java` and
starts `tetris.RlBridge` with a JavaFX-capable JDK (`JAVAFX_JAVA_HOME`, or the
Liberica Java 8 JDK on macOS).

### 2. The environment (`envs/javafx.py`)

`JavaFxControlEnv` starts one bridge process and keeps it across episodes. It
exposes the four-value API used everywhere in the project:

```python
observation = env.reset(seed=42)
observation, reward, done, info = env.step(ControlAction.HARD_DROP)
```

For each frame it decodes the RGB bytes, resizes to 128×96 by nearest-neighbor
index sampling, converts to channel-first `float32` in `[0, 1]`, and appends it
to a four-frame deque. The observation is the four frames concatenated on the
channel axis, so 4 × 3 = 12 channels. On reset the first frame fills all four
slots.

`step()` refuses to run before `reset()` or after a terminal step.

The seven actions are defined in `types.py` as `ControlAction`: no-op, left,
right, rotate clockwise, soft drop, hard drop, hold. The bridge maps them to the
keys a person would press (none, ←, →, ↑, ↓, space, `C`).

### 3. The policy (`models/`)

`SmallCnnEncoder` (`models/encoders.py`) is three convolutions (32×8×8 stride 4,
64×4×4 stride 2, 64×3×3 stride 1, each with ReLU) followed by a linear projection
to 512 features. `VisualActorCritic` (`models/cnn_policy.py`) puts a 7-way
categorical policy head and a scalar value head on the shared features. For the
reference configuration that is 3,244,712 parameters (from the training dry run).

`act()` samples during training and takes the argmax when `deterministic=True`,
which is what evaluation uses.

### 4. Training (`training/visual.py`, `training/ppo.py`)

`train_visual_agent` loops until the step budget is spent. Each iteration
calls three functions:

1. `collect_rollout` plays `rollout_steps` policy steps and returns a `Rollout`
   (observations, actions, log-probabilities, values, rewards, done flags, and
   the episodes that finished). When an episode ends, it resets the environment
   on the next seed (`seed`, `seed + 1`, ...). `EpisodeProgress` carries the
   unfinished episode into the next rollout.
2. `update_policy` bootstraps the value after the last step (`bootstrap_value`,
   zero if that step was terminal), computes advantages and returns with
   `generalized_advantage_estimates`, and runs `ppo_update`: `update_epochs`
   passes over shuffled minibatches with the clipped surrogate, value loss and
   entropy bonus.
3. The loop appends the episode and update rows to `metrics.jsonl` and saves a
   checkpoint after the first update at or past each `checkpoint_interval`
   boundary, plus `final.pt` at the end.

Run outputs go to `experiments/training/<name>-<timestamp>/`
(`resolved_config.json`, `metrics.jsonl`, `summary.json`, `checkpoints/`).

`validate_runtime_support` rejects configuration values the trainer does not
implement yet (other regions, grayscale, macro actions, more than one gravity
tick, other reward kinds) instead of silently running the reference condition.

### 5. Evaluation (`training/evaluate.py`)

`load_visual_checkpoint` rebuilds the model and checks that the checkpoint's
observation, action, model and reward settings match the requested config.
`evaluate_loaded_policy` then plays `evaluation.episodes` episodes on seeds
starting at `evaluation.first_seed`, chooses actions deterministically, and
writes per-episode manifests and JSONL step records (via `events/`) plus
`episodes.json` and `summary.json`. It hashes the model parameters before and
after and fails if they changed. Episodes that hit `max_steps_per_episode` are
marked truncated rather than terminal.

## What the policy can and cannot see

The policy input is pixels only. `reward`, `done` and `info` come from the
game's internal counters. The trainer uses them for the reward, episode
boundaries and logging, and evaluation uses them for metrics. None of them are
added to the observation. Any future condition that feeds internal state to the
model (for example a raw board grid) is a separate, privileged condition and
should be labeled that way.

## Support modules

- `training/config.py`: typed, validated YAML configuration (`load_visual_config`).
  Unknown keys and out-of-range values raise `ConfigurationError`.
- `events/`: a game-neutral episode record format (`manifest.json` plus
  `steps.jsonl`) used by evaluation, `scripts/record_tetris_episode.py` and
  `scripts/replay_tetris_episode.py`.
- `scripts/stability_check.py`: checks the observation contract, same-seed
  determinism and repeated reset/close against a live bridge.

## Placeholders

These files define interfaces for planned comparison conditions but contain no
working implementation. Nothing on the live path imports them.

| Module | State |
|---|---|
| `core/game.py` | `TetrisGame.step_control` and `step_placement` raise `NotImplementedError`; `reset` ignores the seed |
| `envs/control.py`, `envs/placement.py` | Raw-grid environments built on `core/`, so they cannot step yet |
| `observations/raw_grid.py`, `observations/features.py` | Encoders for the privileged comparison conditions |
| `rewards/shaping.py` | Shaped-reward formula; the trainer only supports `sparse_lines` |
| `agents/dqn.py`, `agents/heuristic.py` | Comments only |
| `search/beam.py`, `search/mcts.py` | Comments only |

`agents/random_agent.py` is used by `scripts/random_java_agent.py`.

## Known gaps

- Only the reference condition runs: full UI, RGB, four frames, one gravity tick
  per decision, sparse line reward, PPO.
- Pixel equality across operating systems is not established. Replay has
  `--ignore-pixels` for that reason.
- The Java game state is reached only through the bridge, so the Python
  `core/` engine and the JavaFX game are separate rule implementations. A
  placement or search baseline built on `core/` would have to be checked against
  the JavaFX rules before results are compared.
