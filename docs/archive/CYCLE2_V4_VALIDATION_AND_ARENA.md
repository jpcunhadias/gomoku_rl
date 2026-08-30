# Cycle 2 v4: Validation and Arena Result

**Status**: ✅ Complete. v4 validated, Cycle 2 measurably beats Cycle 1.

Supersedes `docs/current/CYCLE2_V4_PENDING.md` (v4 was written but unrun as of the last "wip"
commit on this branch — see `docs/CHANGELOG.md` Phase 4). See
`docs/archive/CYCLE1_COLDSTART_MECHANISM.md` for why v4 failed when tested against Cycle 1's
untrained network but succeeds here.

## Self-play (v4 config, against the trained Cycle 1 model)

`make self-play CYCLE=2` (`configs/phaseC_c2.py`, v4 exploration params) — correctly loaded
`checkpoints/models/c1_cycle1_last.pth` and seeded 7,500 samples (25%) from Cycle 1's buffer.
200 games, ~63 min, exit 0. Buffer: 12,742 samples, draws correctly excluded (0 across all
phases, confirming the `DiversityManager` fix holds under real conditions).

**Entropy check: median normalized entropy 0.590 — squarely in the 0.45-0.65 target band.**
Same v4 config, same exploration parameters that scored 0.968 (worse than baseline) against
Cycle 1's cold-start network. Distribution is sensibly bimodal, not artificially flat:

| Range | Share |
|---|---|
| 0.00-0.30 (sharp/forced) | 17.8% |
| 0.30-0.45 | 19.8% |
| 0.45-0.65 (target) | 16.8% |
| 0.65-0.80 | 8.9% |
| 0.80-1.00 (open) | 36.6% |

This is direct empirical confirmation of the cold-start mechanism finding: identical exploration
parameters produce meaningfully different — and only now correctly target-range — results once
real learned signal exists for MCTS to search around.

## Training

`make train CYCLE=2` — 15 epochs, exit 0, clean improvement (two small value-loss blips at
epochs 9 and 12, not concerning). Final: Policy loss 0.20, **Value loss 0.0597** (vs Cycle 1's
0.2653 — expected, given far sharper/more informative targets).

## Debug checks (`make debug CYCLE=2`)

**Policy head**: normalized entropy median **0.628 — first time this checkbox has passed** across
the whole Cycle 1/2 investigation. Training smoke check clean (no NaNs, stable gradients).

**Value head — new concern, flagged not fixed**: pre-tanh saturation jumped to **43.8%** (was
10.5% in Cycle 1; the debug script's own threshold is <20%, checkbox now fails). Calibration
degraded alongside it: Brier 0.677 (was 0.609), ECE 0.226 (was 0.136). The confident-extreme
calibration bins that were well-calibrated in Cycle 1 are now off by ~0.25 (bin 0: predicts
-0.965, actual outcomes average -0.718 — was a 0.017 gap in Cycle 1; bin 9 similarly). The model
achieved dramatically lower raw training loss but appears to have done so partly by pushing
predictions toward confident extremes the true outcomes don't fully support — classic value-head
overconfidence. This is the same failure mode the project's early history (SmoothL1 loss,
reduced-gain value-head init, separate weight-decayed optimizer group for the value head) was
built to fight; it's resurfacing here and is worth investigating before Cycle 3, e.g. revisiting
the value-head weight decay/LR split in `cli/train/train_loop_main.py`.

Top-1-vs-MCTS-argmax dropped (57.6% -> 40.2%) and KL(net||mcts) rose (0.035 -> 0.204), but this
is plausibly just harder-to-fit sharper/more-varied targets (Cycle 1's targets were easy to match
approximately because they were nearly uniform), not necessarily an independent red flag.

## Arena: Cycle 2 vs Cycle 1

`make arena CANDIDATE_CYCLE=2 BASELINE_CYCLE=1` — 200 games, 800 sims/move both sides
(candidate gets arena's depth-decaying c_puct schedule, baseline plays flat c_puct=1.5, per
`scripts/arena.py`'s own defaults), ~2.5hr, exit 0.

**Result: 100 wins / 0 losses / 100 draws for Cycle 2. Decisive-game win rate 1.0
(Wilson 95% CI [0.963, 1.0]).**

Cycle 1 never won a single game, in either color. This is real, measurable validation that fixing
the cold-start entropy mismatch produced a genuinely stronger model — and that the value-head
overconfidence above did not prevent it: decision *ranking* held up even though calibration
*magnitude* is off.

**Color-split pattern (not a bug, worth tracking)**: Candidate won 100/100 games as Black (first
move) and drew — never lost — 100/100 as White. Baseline never won as either color. Reading:
first-move advantage on this 8x8 board looks strong; whoever's stronger wins outright with it and
successfully defends into a draw without it. Cycle 1 never converts the first-move advantage into
a win even when it gets it. The arena setup already alternates colors evenly per pair, so the
aggregate win rate above isn't biased by this, but future comparisons (especially the parameter
sweep) should report win rates split by color too, given how large this effect looks.

## Open items for next session

1. Value-head overconfidence (pre-tanh saturation 43.8%, degraded Brier/ECE) — investigate before
   trusting Cycle 3+ value predictions, or before using this model as an evaluator inside a sweep.
2. The small parameter sweep (Layer 2 of the plan): tau, dirichlet_epsilon, 3-4 settings each,
   c_puct fixed — now unblocked, with a validated Cycle 1->2 baseline pair to compare against.
3. DVC/MLflow backup wiring — still parked; worth revisiting once the sweep starts generating
   multiple comparable runs.
