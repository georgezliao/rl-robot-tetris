# Training walkthrough

This page follows one training iteration of the reference visual agent and then
the evaluation of its checkpoint. Each step names the function that does it, the
configuration keys that control it, and the numbers they take in
`configs/visual/reference_ppo.yaml`. For how the pieces connect to the JavaFX
bridge, see [architecture.md](architecture.md).

Line numbers are for commit `336de3a` and will drift. The function names are
the stable reference.

## Setup

[`train_visual_agent`](../src/tetris_rl/training/visual.py#L344) seeds
Python's `random`, NumPy and PyTorch with `seed` (42). It then builds
`VisualActorCritic` and an Adam optimizer (`algorithm.learning_rate`, 2.5e-4),
writes `resolved_config.json` and `provenance.json`, and resets the bridge on
seed 42.

PyTorch's global random generator is used both for sampling actions and for
shuffling minibatches. Any change that draws extra random numbers, such as an
added dropout layer or a different minibatch order, changes every later action,
even with the same seed.

With the reference numbers, one run is 1,000,000 environment steps
(`training.environment_steps`) in rollouts of 128 steps, so 7,813 updates. The
last rollout is 64 steps long.

## 1. Collect a rollout

[`collect_rollout`](../src/tetris_rl/training/visual.py#L161), with
`algorithm.rollout_steps` = 128.

For each step `t` the policy sees the stacked observation `o_t` (shape
`12×128×96`), samples an action `a_t ~ π(·|o_t)`, and records `log π(a_t|o_t)`
and the value estimate `V(o_t)`. The environment returns the next observation,
the raw reward (lines cleared on this step), and `done`.

The stored reward is

```text
r_t = lines_cleared_t + terminal_penalty * done_t
```

with `reward.terminal_penalty` = 0.0 in the reference config, so `r_t` is just
lines cleared ([visual.py:197](../src/tetris_rl/training/visual.py#L197)).

When `done_t` is true, the episode summary goes into `finished_episodes`, and
the environment resets on the next seed through `training_episode_seed`. That
raises `SeedCollisionError` rather than use a seed in the evaluation range
(10000–10099). A rollout can contain the end of one episode and the start of
the next. `EpisodeProgress` carries a partly finished episode into the next
rollout.

The policy runs in `eval()` mode and under `torch.no_grad` (inside
`VisualActorCritic.act`) while collecting, so collection never builds gradients.

## 2. Advantages and returns (GAE)

[`update_policy`](../src/tetris_rl/training/visual.py#L253) first calls
[`bootstrap_value`](../src/tetris_rl/training/visual.py#L242). If the last step
of the rollout was terminal, the value after it is 0. Otherwise it is
`V(o_128)` from the current network.

[`generalized_advantage_estimates`](../src/tetris_rl/training/ppo.py#L94) then
works backward over the rollout, with `algorithm.gamma` (γ = 0.99) and
`algorithm.gae_lambda` (λ = 0.95):

```text
δ_t = r_t + γ (1 − d_t) V_(t+1) − V_t
A_t = δ_t + γ λ (1 − d_t) A_(t+1)
R_t = A_t + V_t
```

`d_t` is 1 on a terminal step. The `(1 − d_t)` factor stops both the value
bootstrap and the advantage recursion at an episode boundary, so episodes
sharing a rollout don't leak into each other. `R_t` is the target for the value
head. `tests/unit/test_gae.py` checks the boundary, bootstrap, λ = 0 and γ = 0
cases against hand-computed values.

## 3. PPO update

[`ppo_update`](../src/tetris_rl/training/ppo.py#L150), controlled by
`update_epochs` (4), `minibatch_size` (64), `clip_epsilon` (0.2),
`value_coefficient` (0.5), `entropy_coefficient` (0.01) and
`max_gradient_norm` (0.5).

Advantages are normalized over the whole rollout once
([ppo.py:176](../src/tetris_rl/training/ppo.py#L176)):
`Â = (A − mean A) / (std A + 1e-8)`. Then, for each of 4 epochs, the 128
transitions are shuffled and split into 2 minibatches of 64. That makes 8
gradient steps per update.

For each minibatch, the network is run again on the stored observations, and
[`ppo_losses`](../src/tetris_rl/training/ppo.py#L40) computes, with
`ρ = exp(log π_new(a|o) − log π_old(a|o))`:

```text
L_policy = −mean( min(ρ Â, clip(ρ, 1 − ε, 1 + ε) Â) )
L_value  = mean( (V_new(o) − R)² )
H        = mean( entropy of π_new(·|o) )
L_total  = L_policy + 0.5 L_value − 0.01 H
```

The `min` makes the objective pessimistic. Once a sample's ratio has moved past
`1 ± ε` in the direction its advantage favors, it stops contributing policy
gradient. `tests/unit/test_ppo_losses.py` works through four samples by hand,
including one clipped above and one clipped below, and checks that the clipped
ones get zero gradient.

The step is `zero_grad`, `L_total.backward()`, gradient-norm clipping to 0.5
([ppo.py:205](../src/tetris_rl/training/ppo.py#L205)), then `optimizer.step()`.
The value head has no separate clipping.

## 4. Logging and checkpoints

After each update the loop in `train_visual_agent` appends rows to
`metrics.jsonl`: one per finished episode, then one for the update.

Episode rows (`"type": "episode"`):

| Field | Meaning |
|---|---|
| `episode` | Training episode index, from 0 |
| `seed` | Piece seed it was played on (`seed + episode`) |
| `environment_steps` | Total training steps when it ended |
| `length` | Steps in the episode |
| `return` | Sum of `r_t` over the episode, including any terminal penalty |
| `score`, `total_lines`, `pieces_placed` | The game's counters at the terminal step, from `info` |

Update rows (`"type": "update"`), averaged over the 8 minibatches of that
update:

| Field | Meaning |
|---|---|
| `update` | Update index, from 1 |
| `environment_steps` | Total training steps after this rollout |
| `policy_loss` | `L_policy` |
| `value_loss` | `L_value` (mean squared error against `R_t`) |
| `entropy` | `H`, in nats. The maximum for 7 actions is ln 7 ≈ 1.95 |
| `approximate_kl` | `mean((ρ − 1) − log ρ)`, an estimate of KL(old ‖ new). Large values mean the policy moved far from the one that collected the data |
| `clip_fraction` | Share of samples with `|ρ − 1| > ε` |

A checkpoint is saved after the first update at or past each multiple of
`training.checkpoint_interval` (50,000). With 128-step rollouts the first one
is `step-50048.pt`, not `step-50000.pt`. `final.pt` is written at the end,
followed by `summary.json` (step, episode and update counts, plus the length
and return of any unfinished episode). Each checkpoint stores the model and
optimizer state, the step count and the full config.

Training reward alone does not show whether the policy got better, and it is not
comparable across reward conditions. Use the evaluation below for that.

## 5. Evaluation

[`load_visual_checkpoint`](../src/tetris_rl/training/evaluate.py#L123)
rebuilds the network and refuses a checkpoint whose observation, action, model
or reward settings differ from the config passed in.
[`evaluate_loaded_policy`](../src/tetris_rl/training/evaluate.py#L255) then
plays `evaluation.episodes` (100) episodes on seeds 10000–10099
(`evaluation.first_seed`). It uses the most likely action (`argmax` of the
logits) instead of sampling. An episode that reaches
`evaluation.max_steps_per_episode` (10,000) is recorded as truncated, not
terminated.

`summary.json` reports, with `N` episodes, `L_i` lines and `P_i` pieces in
episode `i`:

```text
lines_per_100_pieces = 100 · Σ L_i / Σ P_i     (null if Σ P_i = 0)
mean_survival_pieces = Σ P_i / N
```

It also reports mean lines, mean score, and the terminated and truncated
counts. The evaluator hashes the model parameters before and after and records
`parameters_unchanged`. Adding `--results-root results` copies the summary, one
CSV row per episode, and the provenance record into `results/` (see
[results/README.md](../results/README.md)).

One evaluation of one checkpoint from one training seed is a single sample. A
comparison between configurations needs several independently trained seeds on
the same evaluation seeds.
