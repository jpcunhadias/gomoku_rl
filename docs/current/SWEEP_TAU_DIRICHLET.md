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

**Note on the dirichlet axis (44, 46)**: arena'd against Cycle 50 (clean v4 baseline) rather than
Cycle 2, applying the lesson from the tau axis up front instead of needing a correction pass —
Cycle 2's buffer is contaminated (see issue #3 above) and isn't a fair comparison point.

| Cycle | Axis/scale | Ply0 entropy (median) | Ply1 | Ply2 | Held-out Brier | Held-out ECE | Arena result |
|---|---|---|---|---|---|---|---|
| 2 (production, contaminated buffer) | 1.0x both | 0.508 | 0.979 | 0.956 | 0.548 | 0.138 | — (is the baseline) |
| 50 (clean 1.0x re-measure) | 1.0x both | 0.471 | 0.061 | 0.003 | 0.113 | 0.049 | **50-15-35, decisive winrate 77%** vs Cycle 2 — beats it despite ~half the training data (100 vs 200 games), see below |
| 31 | tau 0.75x | 0.381 | 0.008 | 0.000 | 0.124 | 0.046 | **0-49-51, decisive winrate 0.0%** (confirmed under real independent trials; predecessor deterministic run was 0-50-50) |
| 42 | tau 1.25x | 0.570 | 0.096 | 0.008 | 0.144 | 0.036 | **43-0-57, decisive winrate 100%** vs Cycle 2 (confirmed; predecessor deterministic run was 50-0-50, same mirrored color pattern — candidate wins only as White, never loses as Black — persists under real trials, so it's real, not a determinism artifact). **Direct test: beat 50 (clean baseline) 100-0-0, every game.** |
| 61 | tau 1.5x | 0.664 | 0.201 | 0.021 | 0.182 | 0.072 | Not run vs Cycle 2. **Direct test: beat 42 100-0-0, every game.** Trend has not plateaued — 4/4 points so far, each stronger point beats the previous decisively. |
| 44 | dirichlet 0.75x | pending | pending | pending | pending | pending | pending, vs Cycle 50 |
| 46 | dirichlet 1.25x | pending | pending | pending | pending | pending | pending, vs Cycle 50 |

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

### Cycle 50 (clean v4 baseline) — resolves the ply-1/2 puzzle, opens a bigger one

**Ply 1/2 collapse is confirmed real, not a contamination artifact**: Cycle 50's clean readings
(0.061, 0.003) collapse just like both sweep points (31: 0.008, 0.000; 42: 0.096, 0.008). Cycle
2's high ply-1/2 readings (0.979, 0.956) really were entirely the Cycle-1 buffer contamination —
this was the original (correct) small-sample conclusion; the "correction" earlier in this doc's
history was itself wrong, caused by comparing against a contaminated buffer. Ply 0 across all
three clean points (31: 0.381, 50: 0.471, 42: 0.570) is monotonic in tau, exactly as expected.

**Bigger finding**: Cycle 50 and Cycle 2 use the *identical* tau/dirichlet config (v4) — the only
difference is buffer composition (50: 100 games, single-config, clean; 2: 200 games, 25% diluted
with Cycle 1's cold-start data). Cycle 50 **beats** Cycle 2 (50-15-35, 77% decisive win rate)
despite roughly half the training data. Mixing in lower-quality/different-context data hurt more
than the extra volume helped — a real, evidenced data-quality-over-quantity result, and arguably
the most interesting single finding of this sweep so far.

**This also means 31 and 42's wins/losses against Cycle 2 don't cleanly isolate tau's effect** —
Cycle 2 is now shown to be a weaker opponent than a clean v4 model, so comparing sweep points
against it conflates tau's effect with the buffer-purity effect. A clean test of whether higher
tau really does produce a stronger model needs sweep points compared **against each other**
(e.g. 42 vs. 50, or 42 vs. 31), not each against Cycle 2.

### Direct test: 42 vs. 50 (both clean — isolates tau's effect properly)

`make arena CANDIDATE_CYCLE=42 BASELINE_CYCLE=50 ARGS="--games 100 --sims 400"`, stochastic eval.

**Result: 100 wins, 0 losses, 0 draws for Cycle 42 (tau x1.25) — every single game, split evenly
50-50 by color, no draws at all.** The cleanest, most one-sided result in the whole sweep. With
the buffer-purity confound removed (both points are equally clean, single-config buffers), tau
x1.25 is unambiguously stronger than v4's own tau x1.0 in this range — not just "wins more," wins
*every* game regardless of color.

Combined with the rest: 31 (tau x0.75) lost to Cycle 2 (a weaker baseline); 50 (tau x1.0, clean)
beat Cycle 2 decisively; 42 (tau x1.25) beat both Cycle 2 and 50 (the latter perfectly). The
strength ordering **31 < 50 < 42** is now well-supported by direct and indirect evidence alike.
**Tentative conclusion**: within the tested range (0.75x-1.25x of v4), more tau (more stochastic
early-game exploration, specifically at ply 0, since plies 1-2 collapse regardless — see above)
produces a measurably stronger model. Not yet tested: whether this keeps improving beyond 1.25x,
or whether 1.25x is near a ceiling.

### Point 61 (tau 1.5x) — testing whether the trend continues

`configs/sweep_tau_150.py`, cycle 61 (non-adjacent to everything used so far). tau_early_plies
= {0:1.17, 1:0.69, 2:0.42} — note ply 0's exponent (1/1.17 ≈ 0.855) is now *below* 1, an actual
flattening transform rather than sharpening (v4/31/42 were all exponent >1, i.e. some degree of
sharpening). Still the same "scale v4 by X" family, but worth flagging as a slightly different
regime at ply 0 specifically.

Scoped to one arena comparison for this point: **61 vs. 42** (the current strength leader, both
clean buffers) — directly tests whether pushing tau further keeps helping or whether 1.25x is
near a ceiling. Not also run against Cycle 2, to keep compute reasonable; can be added later if
wanted for the results table.

**Result: 100-0-0 for Cycle 61 over Cycle 42 — every game, both colors, no draws.** Same
perfectly one-sided pattern as 42 vs. 50. The trend has not plateaued; if anything it's still
accelerating (Cycle 61's win over 42 is just as total as 42's win over 50).

Ply 0 entropy continues climbing monotonically: 31=0.381, 50=0.471, 42=0.570, **61=0.664**. Ply
1/2 (0.201, 0.021) are higher than 42's but still well below the uniform ceiling — the "always
collapses regardless of tau" read from earlier plies needs an asterisk at this scale; ply 1 in
particular is now clearly responding to tau, not just ply 0.

Held-out calibration: Brier 0.182, ECE 0.072 — **worse than 42's (0.144, 0.036), continuing a
monotonic trend** (50: 0.113 -> 42: 0.144 -> 61: 0.182) in the *opposite* direction from arena
strength, which keeps improving at every step. This mirrors the value-head investigation's
lesson exactly, now showing up cleanly across a whole sweep axis: calibration quality and
playing strength are not the same thing, and here they move in opposite directions as tau
increases. **Open question, not yet answered**: does strength keep climbing past 1.5x, or is a
ceiling close? No sign of one yet in 4 points.
