# Cycle 1: Cold-Start Entropy Mechanism

**Status**: Resolved — explains why Cycle 1's MCTS targets can't be tuned into the 0.45-0.65
normalized-entropy band, and why that's fine.

## Problem

A regenerated Cycle 1 buffer (200 games, `DiversityManager` zero-quota bug already fixed, draws
correctly excluded) still showed median normalized entropy 0.961 — same too-uniform failure mode
as Cycle 2's v2. Seeding Cycle 1's exploration params from Cycle 2's validated v4 balance point
(`configs/phaseC_c1_v4recon.py`, cycle=901 scratch run, 40 games) made it slightly *worse* (0.968),
not better. Tuning tau/Dirichlet further wasn't converging.

## Investigation

Loaded a fresh (untrained) `PolicyValueNet` — the exact state self-play starts from before any
training — and probed its raw outputs on an early-game position:

- **Policy prior**: normalized entropy 0.9989 (1.0 = perfectly uniform) across 62 legal moves.
  Probabilities ranged only 0.0126-0.0199 (mean 0.0161, std 0.0016) — essentially flat.
- **Value estimates**: across 20 different one-ply-ahead candidate positions, values spanned only
  -0.0129 to -0.0006 (std 0.0026) — indistinguishable from noise around zero.

## Root cause

MCTS/PUCT concentrates visits by exploiting differences in `Q` (value) and `P` (prior). With both
this flat, there is no real signal for the search to concentrate on. No choice of tau or Dirichlet
noise can produce a genuinely sharp target, because sharpening requires something underneath to
sharpen *around*. The 0.45-0.65 target-entropy band is a property of a network with real learned
signal — not something exploration-parameter tuning can manufacture on random weights.

This reframes several things that looked like separate mysteries:

- **Cycle 2's v2->v3 flip** (0.973 too-uniform straight to 0.196 too-sharp, no stable middle):
  both are noise, just amplified differently. Low tau doesn't sharpen around a meaningful peak —
  it crowns whichever child randomly accumulated a few extra visits by chance. Confidently
  arbitrary, not confidently correct. (Note: Cycle 2's v1-v4 tuning was legitimate and meaningful,
  because it ran against an already-*trained* Cycle 1 model, not a cold-start one — see below.)
- **v4 seeding making Cycle 1 worse**: tuning noise, not signal.
- **The original Cycle 1 bug** (raw entropy *collapse*, opposite direction, `docs/CHANGELOG.md`
  Cycle 1 section): a near-zero effective tau from the `tau_early_plies` key-mismatch bug turned
  noisy near-uniform visit counts into a falsely confident, essentially arbitrary pick. Same
  underlying noise, different artifact.

## Conclusion / decision

Stopped tuning Cycle 1's exploration params. Trained on the existing Cycle 1 buffer as-is (entropy
~0.96 is expected and correct for cold-start data, not a defect). The entropy gate is the right
diagnostic starting at **Cycle 2**, once self-play runs against a trained model with real Q/P
differentiation — see `docs/archive/CYCLE2_V4_VALIDATION_AND_ARENA.md`, where the identical v4
config that failed here produced median normalized entropy 0.590, squarely in-band, on the exact
same knobs.

This is worth keeping as a general note for any future from-scratch cycle (a real restart, a new
board size, a new architecture): don't gate self-play data quality on target-entropy checks until
there's a trained model generating the priors/values MCTS searches with. Cycle 1's own data quality
should instead be judged by whether *training* on it produces a healthy, improving loss curve
(it did — see the Cycle 1 debug results below), not by whether its raw self-play entropy looks
"sharp."

## Cycle 1 outcome (for the record)

- Buffer: 10,445 samples (200 games, draws correctly excluded post-fix)
- Trained 10 epochs, exit 0, clean monotonic improvement: Policy loss 3.75->1.31, Value loss
  0.466->0.265
- Debug checks (`make debug CYCLE=1`): value head shows real calibration (Brier 0.609, ECE 0.136,
  confident-extreme bins well calibrated e.g. -0.914 predicted vs -0.897 actual, saturation 10.5%
  well under the 20% threshold), policy head faithfully learned the noisy targets (KL(net||mcts)
  median 0.000, top-1 agreement with MCTS argmax 57.6%) — a training issue would show as *high*
  KL against known-good targets; this is a data-quality ceiling, correctly learned.
