# Sweep: tau and dirichlet_epsilon around Cycle 2's v4 config

**Status**: In progress. Turned out to be as much a methodology audit as a parameter sweep —
three real issues in existing tooling were found and fixed while running it. See "Issues found
and fixed" below before trusting any single number in the results table in isolation.

## Design

One-factor-at-a-time (not full factorial) around Cycle 2's v4 config as the shared center point.
`c_puct` stays fixed throughout.

| Cycle | Axis | Scale | tau_early_plies | dirichlet_epsilon / root |
|---|---|---|---|---|
| 2 (production) | both | 1.0x | {0:0.78, 1:0.46, 2:0.28} | 0.10 / 0.50 |
| 50 | both (clean re-measure) | 1.0x | {0:0.78, 1:0.46, 2:0.28} | 0.10 / 0.50 |
| 31 | tau | 0.75x | {0:0.585, 1:0.345, 2:0.21} | 0.10 / 0.50 (fixed) |
| 42 | tau | 1.25x | {0:0.975, 1:0.575, 2:0.35} | 0.10 / 0.50 (fixed) |
| 44 | dirichlet | 0.75x | {0:0.78, 1:0.46, 2:0.28} (fixed) | 0.075 / 0.375 |
| 46 | dirichlet | 1.25x | {0:0.78, 1:0.46, 2:0.28} (fixed) | 0.125 / 0.625 |

Configs: `configs/sweep_v4_baseline_clean.py`, `configs/sweep_tau_075.py`, `configs/sweep_tau_125.py`,
`configs/sweep_dirichlet_075.py`, `configs/sweep_dirichlet_125.py`.

**Reduced compute for a faster first pass**: self-play 100 games (not 200), arena 100 games /
400 sims (not 200/800). If a setting looks promising, worth a full-size confirmation run before
trusting it further.

## Issues found and fixed while running this sweep

Three, in the order discovered — each changes how to read results that came before the fix.

### 1. Cycle-numbering collision (fixed before it cost anything)

`cycle_paths(N-1)` is the self-play buffer-seeding fallback, so sequential cycle numbers
(31/32/33/34) made sweep point N+1 silently inherit 25% of sweep point N's buffer as "the
previous cycle" — exactly the dilution the design is trying to avoid. Caught within seconds of
launching point 2, killed and cleaned up before it wasted compute, renumbered to non-adjacent
ids (31, 42, 44, 46, 50).

### 2. Arena was replaying one deterministic game N times

`scripts/arena.py` defaults to `temperature=0.0` and no Dirichlet noise (fully deterministic)
unless `--stochastic_eval` is passed, which the Makefile's `arena` target never did — and there's
no opening variety either. Given the same two models and the same empty starting board, every
"game" within a color assignment produced identical moves. Every arena result all session
(Cycle 1 vs 2, and sweep points 31 and 42) was actually just 2 unique games (one per color)
replayed dozens of times — explains the suspiciously perfect all-or-nothing splits in every
result, and means the reported Wilson 95% CIs were built on an effective n of 2, not 50-200.

**Fixed**: `--stochastic_eval` is now on by default in the Makefile's `arena` target (root
Dirichlet noise + small temperature at plies 0-1, deterministic after — already existed in
`scripts/arena.py`, just was never turned on). **Points 31 and 42's arena results in the table
below predate this fix and are being rerun** (`logs/arena_rerun_stochastic.log` on the server).
Cycle 1 vs 2's original headline arena result (`docs/archive/CYCLE2_V4_VALIDATION_AND_ARENA.md`)
has the same problem and hasn't been rerun yet — treat its confidence framing as overstated until
it is.

### 3. Entropy check pooled all plies into one misleading number, AND Cycle 2 isn't a clean 1.0x sample

Two compounding problems in `debug/check_mcts_target_entropy.py`'s original pooled median:

- **Pooling across plies.** `tau_early_plies` only applies to plies 0-2; ply 0 is where tau
  clearly matters, but plies 1-2 are far more sensitive to the specific self-play run (see next
  point) and pooling them into one median can hide or misrepresent what's happening at ply 0.
  Fixed: the script now reports per-ply stats (ply is recovered from the state tensor's stone
  count, not stored separately in the buffer) alongside the pooled number, which is kept for
  backward-compat with the 0.45-0.65 target used throughout `docs/CHANGELOG.md`. Also fixed:
  the original script capped its sample at 2048 out of a much larger buffer, badly undersampling
  the inherently small pool of early-ply positions (~1 per game); it now uses the whole buffer,
  which as a side effect also made results reproducible run-to-run (they weren't before).
- **Cycle 2's buffer isn't a clean sample of v4's config.** With the per-ply, full-buffer version
  of the check, Cycle 2 shows *much* higher ply-1/2 entropy (0.979, 0.956) than either sweep
  point (31: 0.008, 0.000; 42: 0.096, 0.008) — not something tau alone explains, since 42 has a
  *higher* tau than v4 at those plies and still collapsed. Reason: Cycle 2's buffer seeded 25%
  of itself from Cycle 1's buffer (cold-start, untrained-network self-play, which has elevated
  entropy at *every* ply for unrelated reasons — see `docs/archive/CYCLE1_COLDSTART_MECHANISM.md`).
  The sweep points are deliberately unseeded, clean, single-config buffers. Comparing them against
  Cycle 2's mixed buffer was comparing a clean measurement to a contaminated one.

**Fixed**: added `configs/sweep_v4_baseline_clean.py` (cycle 50) — identical tau/dirichlet to
Cycle 2, but generated with the same methodology as the other sweep points (unseeded buffer),
for a genuinely fair 1.0x reference point. Queued to run after the arena reruns finish. This
does *not* affect the arena results — those test Cycle 2's actual trained model, a real artifact
regardless of its buffer's composition — only the entropy comparison.

## Seeding methodology (applies to all sweep points, including 50)

Each sweep cycle's self-play searches with **Cycle 2's trained weights** (copied into
`checkpoints/models/c1_cycle{N}_last.pth` before self-play runs), matching the project's
established finding that self-play entropy/quality diagnostics only mean something against a
network with real learned signal (`docs/archive/CYCLE1_COLDSTART_MECHANISM.md`).

Each sweep cycle **starts with an empty buffer** — deliberately *not* seeded from any other
buffer (see issue #3 above for why this matters).

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

`make arena` now includes `--stochastic_eval` by default (see issue #2), no extra ARGS needed
for that.

## Results

Fill in / correct as each point (re)completes with the fixed methodology.

| Cycle | Axis/scale | Ply0 entropy (median) | Ply1 | Ply2 | Held-out Brier | Held-out ECE | Arena vs Cycle 2 (W-L-D, decisive) |
|---|---|---|---|---|---|---|---|
| 2 (production, contaminated buffer) | 1.0x both | 0.508 | 0.979 | 0.956 | 0.548 | 0.138 | — (is the baseline) |
| 50 (clean 1.0x re-measure) | 1.0x both | pending | pending | pending | pending | pending | pending |
| 31 | tau 0.75x | 0.381 | 0.008 | 0.000 | 0.124 | 0.046 | **0-49-51, decisive winrate 0.0%** (confirmed under real independent trials; predecessor deterministic run was 0-50-50) |
| 42 | tau 1.25x | 0.570 | 0.096 | 0.008 | 0.144 | 0.036 | **43-0-57, decisive winrate 100%** (confirmed; predecessor deterministic run was 50-0-50, same mirrored color pattern — candidate wins only as White, never loses as Black — persists under real trials, so it's real, not a determinism artifact) |
| 44 | dirichlet 0.75x | | | | | | |
| 46 | dirichlet 1.25x | | | | | | |

### Points 31 and 42 — confirmed under real independent trials

Both arena reruns landed with genuinely varying results (43-57 / 49-51 splits, not the old
deterministic 50-50), confirming the games really are independent now, and both strength
conclusions hold: **tau x0.75 (point 31) is robustly weaker** (0% decisive win rate), **tau x1.25
(point 42) is robustly stronger** (100% decisive win rate) than Cycle 2. Point 42's mirrored color
pattern (wins only as White, never loses as Black) also persists under real trials, so it's a real
effect, not a determinism artifact — no mechanistic explanation for it yet.

Point 31's story: ply 0 entropy (0.381) is clearly below v4's contaminated-buffer reading (0.508),
in the direction theory predicts for less exploration; narrower exploration means self-play
visited a much narrower slice of the game tree with high confidence, producing a model that
predicts outcomes very accurately *for that narrow, homogeneous set of positions* (better Brier/
ECE than Cycle 2) without that transferring to strength against a genuinely different opponent.

Point 42's ply 0 entropy (0.570) is *above* v4's reading, in the other direction. Both 31 and 42
collapsed similarly at plies 1-2, yet one is weaker and the other stronger — so ply-1/2 collapse
alone doesn't explain strength; ply 0 is the one axis that tracks the strength result cleanly.
**Held loosely until Cycle 50 lands**, since "v4's ply-0 entropy" is still the contaminated 0.508
reading, and it's not yet confirmed whether the Cycle-1 contamination affected ply 0 as much as it
clearly affected plies 1-2 (see issue #3 above).
