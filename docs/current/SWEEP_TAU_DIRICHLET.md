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
| 42 | tau | 1.25x | {0:0.975, 1:0.575, 2:0.35} | 0.10 / 0.50 (fixed) |
| 44 | dirichlet | 0.75x | {0:0.78, 1:0.46, 2:0.28} (fixed) | 0.075 / 0.375 |
| 46 | dirichlet | 1.25x | {0:0.78, 1:0.46, 2:0.28} (fixed) | 0.125 / 0.625 |

**Cycle numbers are deliberately non-adjacent** (31, 42, 44, 46 — not 31/32/33/34). Discovered the
hard way: `cycle_paths(N-1)` is used as the self-play buffer-seeding fallback, so sequential
numbers make sweep point N+1 silently inherit 25% of sweep point N's buffer as "the previous
cycle" — exactly the dilution this design is trying to avoid. Point 32 (tau 1.25x) was launched,
caught seeding from point 31's buffer within seconds, killed and cleaned up before it wasted
compute, then relaunched as cycle 42. Keep future sweep cycle numbers non-adjacent to each other
(and to any real cycle N-1) for this reason.

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
uv run python debug/check_mcts_target_entropy.py --buffer checkpoints/buffers/replay_c1_cycle{N}.pkl
uv run python scripts/diagnose_value_head_holdout.py \
    --buffer checkpoints/buffers/replay_c1_cycle{N}.pkl \
    --init_checkpoint checkpoints/models/c1_cycle2_last.pth
make arena CANDIDATE_CYCLE={N} BASELINE_CYCLE=2 ARGS="--games 100 --sims 400"
```

## Results

Fill in as each point completes.

| Cycle | Axis/scale | Normalized entropy (median, MCTS target) | Held-out Brier | Held-out ECE | Held-out sat% | Arena vs Cycle 2 (W-L-D, decisive) |
|---|---|---|---|---|---|---|
| 2 (baseline) | 1.0x both | 0.590 | 0.548 | 0.138 | 40.0% | — (is the baseline) |
| 31 | tau 0.75x | **0.045** (severe over-sharp collapse) | 0.124 | 0.046 | 84.9% | **0-50-50 (0% decisive win rate)** |
| 42 | tau 1.25x | | | | | |
| 44 | dirichlet 0.75x | | | | | |
| 46 | dirichlet 1.25x | | | | | |

### Point 31 (tau 0.75x) — read

A striking, coherent result: this model's calibration metrics *look better* than Cycle 2's
(lower Brier, lower ECE, both on training data and on a genuine held-out split) — but it lost
every single decisive arena game against Cycle 2 (0 wins, 50 losses, 50 draws; same color pattern
as the Cycle 1 vs 2 arena — the stronger side only won as Black, never lost even without the
first-move advantage). The explanation: tau x0.75 collapsed self-play into the same over-sharp
failure mode as Cycle 2's v3 (normalized entropy 0.045, matching v3's 0.196 in kind if not degree)
— reduced exploration means self-play visited a much narrower slice of the game tree with high
confidence, so the model learned to predict outcomes very accurately *for that narrow, homogeneous
set of positions* (hence the flattering Brier/ECE), but that doesn't transfer to actual strength
against a genuinely different, more broadly-competent opponent. **Calibration metrics alone can't
tell you if an exploration setting is good — you need the arena test.** This is exactly why the
sweep measures both.
