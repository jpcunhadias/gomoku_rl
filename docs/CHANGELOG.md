# Cycle Changelog

This document consolidates the history of all training cycles, their issues, fixes, and outcomes.

## Cycle 1

### Status: Completed (with issues)

### Problem Identified
- **Critical Issue**: Exploration collapsing at ply 0
- 61.3% of first moves had entropy < 1.0 (target: >2.0)
- Median entropy: 0.78 (target: >2.0)
- 53.3% of games had early entropy < 1.5

### Root Causes
1. **Bug**: `tau_early_plies` dict not applied due to int/str key mismatch after JSON serialization
2. **Config**: Parameters too conservative (tau=0.55, epsilon_root=0.40)
3. **Logging**: `dirichlet_eps` logging showed wrong value

### Solutions Applied
1. Fixed `_tau_for_move()` to handle both int and str keys
2. Fixed `dirichlet_eps` logging to show actual epsilon used
3. Created `phaseC_c2.py` with more aggressive exploration parameters

### Key Learnings
- JSON serialization converts dict int keys to strings
- Need to handle both key types in config loading
- Exploration parameters need to be more aggressive than initially thought

### Addendum: regenerated on a fresh server, cold-start entropy mechanism found
Re-run from scratch on a new server with no prior checkpoints (200 games, `DiversityManager`
zero-quota bug fixed — see below). Median normalized entropy 0.961, same too-uniform pattern as
Cycle 2 v2, and seeding Cycle 2's v4 exploration params onto Cycle 1 made it slightly *worse*
(0.968). Investigated why instead of continuing to guess parameters: a fresh untrained network's
policy prior and value estimates are both nearly flat (prior normalized entropy 0.9989, value
estimates across 20 candidate moves span only -0.0129 to -0.0006). With no real Q/P signal, no
exploration-parameter tuning can produce a genuinely sharp target — the 0.45-0.65 entropy target
is only meaningful once a network has real learned signal, i.e. from Cycle 2 onward, not for
Cycle 1's own cold-start self-play. Trained on the Cycle 1 buffer as-is instead of chasing this
further. Full writeup: `docs/archive/CYCLE1_COLDSTART_MECHANISM.md`.

---

## Cycle 2

### Status: ✅ Completed — v4 validated, beats Cycle 1 in arena (100W-0L-100D)

### Phase 1: Initial Success (v1)
- **Achievement**: Excellent exploration metrics achieved
- Ply 0 entropy: 3.18 (target: >2.0) ✅
- Early game entropy: 3.998 (target: >2.0) ✅
- All exploration targets exceeded

### Phase 2: Training Issue (v2)
- **Problem**: Model learned overly uniform policies
- Normalized entropy: 0.969 (target: 0.45-0.65) ⚠️
- **Initial Solution**: Increased training (epochs: 10→15, steps: 50→60)
- **Result**: Still high normalized entropy after retraining

### Phase 3: Root Cause Identified (v3)
- **Investigation**: Checked MCTS target policies in training data
- **Finding**: MCTS target policies have normalized entropy 0.973 (too uniform!)
- **Root Cause**: Exploration parameters too high → uniform MCTS policies → uniform model
- **Solution**: Reduced exploration parameters:
  - tau_early_plies[0]: 0.85 → 0.70
  - dirichlet_epsilon_root: 0.55 → 0.45
  - dirichlet_epsilon: 0.12 → 0.08
  - c_puct_schedule c0: 0.8 → 0.6

### Key Learnings
- **Normalized entropy** measures policy sharpness, not exploration
- **Raw entropy** measures exploration diversity
- Need to balance both: good exploration (raw >2.0) AND sharp policies (normalized 0.45-0.65)
- Model correctly learns from training data - if data is uniform, model will be uniform
- Must check MCTS target policies, not just model policies

### Phase 4: Overcorrection Found (v3 results) and Rebalance (v4)
- **v3 was run** (self-play regenerated with the reduced-exploration config, retrained)
- **Finding**: v3 overcorrected — normalized entropy 0.196 (target 0.45-0.65, now too *sharp*),
  raw entropy 0.805 (target >2.0, exploration collapsed again)
- ⚠️ **Provenance note**: these v3 numbers only exist as a code comment in the `phaseC_c2.py`
  docstring (added when v4 was written). No dedicated validation report or debug-check output
  was ever committed for v3, unlike v1/v2. Treat them as directionally correct, not verified —
  worth re-running `make debug` against the v3 buffer/checkpoint if they still exist on the
  server, to get a real report before trusting them as a sweep data point.
- **Solution (v4)**: split the difference between v2 (too uniform) and v3 (too sharp):
  - tau_early_plies[0]: 0.70 → 0.78
  - tau_early_plies[1]: 0.42 → 0.46
  - dirichlet_epsilon_root: 0.45 → 0.50
  - dirichlet_epsilon: 0.08 → 0.10
  - c_puct_schedule c0: 0.6 → 0.65
- **Status**: config written (`configs/phaseC_c2.py`), **never run**. This is the actual
  frontier of the project as of the last "wip" commit (2026-01-20) — everything above this line
  happened; nothing below it has.

### Phase 5: v4 run against a trained Cycle 1 model — validated
Self-play with v4 config, now against the *trained* Cycle 1 model (not a fresh network) and
seeded 25% from Cycle 1's buffer: 200 games, 12,742 samples, draws correctly excluded.
**Median normalized entropy 0.590 — in the 0.45-0.65 target band**, on the identical config that
scored 0.968 against Cycle 1's cold-start network (see the Cycle 1 addendum above). Direct
confirmation of the cold-start mechanism finding.

Trained 15 epochs (exit 0, value loss 0.0597). Debug checks: policy head normalized entropy
median 0.628 — **first time this checkbox has passed**. But value head shows new overconfidence:
pre-tanh saturation 43.8% (was 10.5%, threshold <20%), Brier 0.677 (was 0.609), ECE 0.226 (was
0.136) — flagged, not yet fixed.

**Arena vs. Cycle 1**: 200 games, 800 sims/move, ~2.5hr. **100 wins / 0 losses / 100 draws,
decisive win rate 1.0 (Wilson95 [0.963, 1.0])**. Cycle 1 never won a single game. Notable color
pattern: candidate won 100/100 as Black, drew (never lost) 100/100 as White — first-move
advantage looks strong on this board, and Cycle 1 never converts it even when it gets it.

Full writeup: `docs/archive/CYCLE2_V4_VALIDATION_AND_ARENA.md`.

### Current Status
- Cycle 2 is done and validated. See "Next Steps" below for what's open.

---

## Configuration Evolution

### Cycle 1 → Cycle 2 (v1)
- Increased exploration parameters significantly
- Fixed bugs in config handling
- Result: Excellent exploration, but too uniform policies

### Cycle 2 (v1) → Cycle 2 (v2)
- Increased training parameters
- Result: Still uniform policies (data issue, not training issue)

### Cycle 2 (v2) → Cycle 2 (v3)
- Reduced exploration parameters moderately
- Result: overcorrected — too sharp, exploration collapsed (see Phase 4; unverified provenance)

### Cycle 2 (v3) → Cycle 2 (v4)
- Split the difference between v2 and v3 on every knob
- Goal: raw entropy >2.0 AND normalized entropy 0.45-0.65 at the same time
- Run against the trained Cycle 1 model: **succeeded** (median normalized entropy 0.590)

---

## Best Practices Established

1. **Always check MCTS target entropy** when debugging policy issues
2. **Distinguish between raw entropy (exploration) and normalized entropy (sharpness)**
3. **Model learns from data** - if data is uniform, model will be uniform
4. **Version configs** when making significant changes
5. **Keep cycle checkpoints** for seeding future cycles
6. **Document issues and solutions** for future reference
7. **The entropy gate only means something once a network has real learned signal** — don't
   apply it to a from-scratch cycle's own cold-start self-play (see the Cycle 1 addendum)
8. **Report win rates split by color, not just aggregate**, in any arena comparison — first-move
   advantage looks strong on this board (see Cycle 2 Phase 5)
9. **Evaluate calibration on a held-out split, not the training buffer** — comparing two models'
   calibration on *their own* training buffers isn't a controlled test; different buffers have
   different difficulty. Use `scripts/diagnose_value_head_holdout.py`. See
   `docs/archive/VALUE_HEAD_CALIBRATION_INVESTIGATION.md`.

---

## Next Steps

1. ~~Value-head overconfidence~~ **Investigated and closed** — didn't survive a controlled
   held-out test (training on Cycle 2 improved Brier 0.816→0.548 and ECE 0.178→0.138 on genuinely
   unseen positions; the original cross-buffer comparison that flagged this was methodologically
   flawed). See `docs/archive/VALUE_HEAD_CALIBRATION_INVESTIGATION.md`.
2. ~~The small parameter sweep~~ **Run, but every result predates a real arena confound found
   during a later soundness audit** — `scripts/arena.py` hard-coded the candidate to always get
   a search-time c_puct bonus the baseline never got, and the model under test was always the
   candidate. Fixed (`--candidate_schedule` now defaults to `False`, matching `--baseline_schedule`),
   but nothing has been rerun under the fix yet. **Rerunning the sweep under the fix is the
   actual next step**, not starting something new — see `docs/current/SWEEP_TAU_DIRICHLET.md`.
3. **DVC/MLflow backup wiring** — still parked; revisit once the sweep starts producing multiple
   comparable runs worth not losing.
4. **The XAI layer** — waiting on arena results being trustworthy again before building
   explainability tooling on top of "which model is actually stronger."

