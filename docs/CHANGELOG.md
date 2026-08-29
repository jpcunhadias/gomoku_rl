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

---

## Cycle 2

### Status: In Progress (v4 config written, not yet run)

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

### Current Status
- v4 config is ready in `configs/phaseC_c2.py`
- Not yet run: regenerate Cycle 2 self-play with v4, retrain, re-check MCTS target entropy
- Expected: MCTS normalized entropy → 0.45-0.65 AND raw entropy >2.0 simultaneously

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
- Not yet run

---

## Best Practices Established

1. **Always check MCTS target entropy** when debugging policy issues
2. **Distinguish between raw entropy (exploration) and normalized entropy (sharpness)**
3. **Model learns from data** - if data is uniform, model will be uniform
4. **Version configs** when making significant changes
5. **Keep cycle checkpoints** for seeding future cycles
6. **Document issues and solutions** for future reference

---

## Next Steps

- Regenerate Cycle 2 self-play with v4 config (`configs/phaseC_c2.py`)
- Run `debug/check_mcts_target_entropy.py` and validate normalized entropy is in the
  0.45-0.65 range *and* raw entropy is >2.0 (both, not just one — see Phase 4)
- Train model and verify normalized entropy holds after training
- Mark Cycle 2 as successful when targets met; only then move `docs/current/` → `docs/archive/`

