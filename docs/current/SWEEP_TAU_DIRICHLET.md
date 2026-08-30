# Sweep: tau and dirichlet_epsilon around Cycle 2's v4 config

**Status**: In progress.

## Design

One-factor-at-a-time (not full factorial) around Cycle 2's v4 config as the shared center point.
`c_puct` stays fixed throughout. Cycle 2 itself *is* the 1.0x point for both axes, so it's reused
rather than rerun — 4 new runs total, not 6-8.

| Cycle | Axis | Scale | tau_early_plies | dirichlet_epsilon / root |
|---|---|---|---|---|
| 2 (reused) | both | 1.0x | {0:0.78, 1:0.46, 2:0.28} | 0.10 / 0.50 |
| 31 | tau | 0.75x | {0:0.585, 1:0.345, 2:0.21} | 0.10 / 0.50 (fixed) |
| 32 | tau | 1.25x | {0:0.975, 1:0.575, 2:0.35} | 0.10 / 0.50 (fixed) |
| 33 | dirichlet | 0.75x | {0:0.78, 1:0.46, 2:0.28} (fixed) | 0.075 / 0.375 |
| 34 | dirichlet | 1.25x | {0:0.78, 1:0.46, 2:0.28} (fixed) | 0.125 / 0.625 |

Configs: `configs/sweep_tau_075.py`, `configs/sweep_tau_125.py`, `configs/sweep_dirichlet_075.py`,
`configs/sweep_dirichlet_125.py`.

**Reduced compute for a faster first pass**: self-play 100 games (not 200), arena 100 games /
400 sims (not 200/800). If a setting looks promising, worth a full-size confirmation run before
trusting it further.

## Seeding methodology (important, not the default cycle-chain behavior)

Each sweep cycle's self-play searches with **Cycle 2's trained weights** (copied into
`checkpoints/models/c1_cycle{N}_last.pth` before self-play runs), matching the project's
established finding that self-play entropy/quality diagnostics only mean something against a
network with real learned signal (`docs/archive/CYCLE1_COLDSTART_MECHANISM.md`).

Each sweep cycle **starts with an empty buffer** — deliberately *not* seeded from Cycle 2's
buffer (unlike the normal cycle-to-cycle 25%-tail-seeding pattern). Mixing in Cycle 2's ~12.7k
samples would dilute a 100-game (~6k sample) sweep buffer with data from a different exploration
setting, blurring exactly the effect the sweep is trying to isolate. So each sweep point trains
on a small, clean, self-contained buffer purely from its own tau/dirichlet setting.

`best_value_loss` tracking also starts fresh per sweep point (the copied `_last.pth` doesn't
carry that field), so "best checkpoint" reflects this run's own training, not inherited from
Cycle 2's differently-scaled loss.

## Per-point pipeline

```bash
cp checkpoints/models/c1_cycle2_last.pth checkpoints/models/c1_cycle{N}_last.pth
make self-play CYCLE={N} CONFIG=sweep_{name}
make train CYCLE={N} CONFIG=sweep_{name}
make debug CYCLE={N}
uv run python scripts/diagnose_value_head_holdout.py \
    --buffer checkpoints/buffers/replay_c1_cycle{N}.pkl \
    --init_checkpoint checkpoints/models/c1_cycle2_last.pth
make arena CANDIDATE_CYCLE={N} BASELINE_CYCLE=2 ARGS="--games 100 --sims 400"
```

## Results

Fill in as each point completes.

| Cycle | Axis/scale | Raw entropy (ply0) | Normalized entropy (median) | Held-out Brier | Held-out ECE | Arena vs Cycle 2 (W-L-D) |
|---|---|---|---|---|---|---|
| 2 (baseline) | 1.0x both | — | 0.590 | 0.548 | 0.138 | — (is the baseline) |
| 31 | tau 0.75x | | | | | |
| 32 | tau 1.25x | | | | | |
| 33 | dirichlet 0.75x | | | | | |
| 34 | dirichlet 1.25x | | | | | |
