# Results

Each directory here is the committed summary of one evaluation run. Tables in
the main README should cite numbers only from these directories.

Directory names are `<date>-<device>-<evaluation run>`, for example
`2026-10-30-apple-m3-visual_reference_ppo-20261030T120000.000000Z`. The last
part matches the full run under `experiments/evaluation/`, which stays on the
machine that produced it and is ignored by Git.

| File | Contents |
|---|---|
| `summary.json` | Aggregate metrics: lines, score, survival in pieces, lines per 100 pieces, terminated and truncated counts, seeds, checkpoint SHA-256 and step count |
| `episodes.csv` | One row per evaluation episode: seed, steps, terminated/truncated, return, score, lines, pieces |
| `provenance.json` | Repository and `tetris/` commits with dirty flags, config path, seeds, device, OS, software versions, checkpoint path and hash |

What stays in `experiments/` and out of Git: checkpoints, per-step JSONL
records, episode manifests, training metrics and logs.

To produce a results directory, pass `--results-root results` to the evaluation
command:

```bash
PYTHONPATH=src python -m tetris_rl.training.evaluate \
  --config configs/visual/reference_ppo.yaml \
  --checkpoint experiments/training/<run>/checkpoints/final.pt \
  --results-root results
```

To get a learning curve for a training run, evaluate all of its checkpoints:

```bash
TETRIS_BRIDGE_OFFSCREEN=1 PYTHONPATH=src python -m tetris_rl.training.sweep \
  --config configs/visual/reference_ppo_terminal.yaml \
  --run-directory experiments/training/<run>
```

This writes `results/<date>-<device>-curve-<run>/learning_curve.csv` (one row
per checkpoint: steps, mean survival, lines per 100 pieces, mean lines) and
`provenance.json`. The curve shows progress during training. The reported
result is `final.pt`, not the best point on the curve.

A run made with `--episodes` or `--max-steps-per-episode` overrides is a
diagnostic; its `summary.json` records the seeds and step limit actually used.
Check `provenance.json` before comparing two directories: results from
different machines, commits with uncommitted changes, or different seed ranges
are not directly comparable.
