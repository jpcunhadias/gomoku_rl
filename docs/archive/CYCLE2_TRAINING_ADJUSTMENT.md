# Cycle 2 Training Adjustment

**Date**: After debug check revealed high normalized entropy
**Status**: ✅ Config updated, ready to retrain

## Problem Identified

After Cycle 2 training completed, debug check revealed:
- ✅ Excellent exploration metrics (entropy 3.18 at ply 0)
- ✅ Model training stable (losses decreasing, no NaNs)
- ⚠️ **Normalized entropy: 0.969** (target: 0.45-0.65) - TOO HIGH
- Model learned overly uniform/exploratory policies

## Root Cause Analysis

The Cycle 2 self-play data is **excellent** - exploration metrics exceeded all targets:
- Ply 0 entropy: 3.18 (target: >2.0) ✅
- Early game entropy: 3.998 (target: >2.0) ✅

However, the model didn't train long enough to learn sharper policies from this good exploration data. The training was stopped at 10 epochs, which wasn't sufficient to learn decisive policies.

## Solution: Option A - Increase Training Only

**Decision**: Keep the excellent Cycle 2 self-play data, increase training parameters.

### Changes Applied to `phaseC_c2.py`

| Parameter | Before | After | Change | Rationale |
|-----------|--------|-------|--------|-----------|
| `epochs` | 10 | **15** | +50% | More training to learn sharper policies |
| `steps_per_epoch` | 50 | **60** | +20% | More steps per epoch for better learning |

### What Was NOT Changed

- ✅ Self-play exploration parameters (already excellent)
- ✅ Buffer data (23,719 samples from 377 games)
- ✅ All other training hyperparameters

## Files Modified

- ✅ `configs/phaseC_c2.py` - Updated training parameters
- ✅ `checkpoints/models/c1_cycle2_*.pth` - Deleted (will be regenerated)

## Files Preserved

- ✅ `checkpoints/buffers/replay_c1_cycle2.pkl` - Kept (excellent data)
- ✅ `checkpoints/selfplay/selfplay_c1_cycle2.jsonl` - Kept (excellent exploration)
- ✅ All other Cycle 2 artifacts preserved

## Next Steps

### 1. Retrain Cycle 2

```bash
make train CYCLE=2
```

This will:
- Load the existing buffer (23,719 samples)
- Train for 15 epochs (instead of 10)
- Use 60 steps per epoch (instead of 50)
- Save new model checkpoints

### 2. Validate Results

After training completes:

```bash
make debug CYCLE=2
```

Check for:
- ✅ Normalized entropy should decrease toward 0.45-0.65 range
- ✅ Policy losses should continue decreasing
- ✅ Model should learn sharper policies

### 3. Evaluate Performance

```bash
# Compare with Cycle 1 (if exists)
make arena CANDIDATE_CYCLE=2 BASELINE_CYCLE=1

# Or evaluate vs pure MCTS
python scripts/arena.py \
  --candidate checkpoints/models/c1_cycle2_best.pth \
  --baseline checkpoints/baselines/phaseB_proxy.pth \
  --games 200 --sims 800
```

## Expected Impact

### Training
- **More training time**: 15 epochs × 60 steps = 900 total steps (vs 500 before)
- **80% more training steps** should help model learn sharper policies

### Policy Learning
- Normalized entropy should decrease from 0.969 toward target range (0.45-0.65)
- Model should learn more decisive policies
- Top-1/Top-3 alignment may improve

### Exploration (Should Remain Excellent)
- Self-play data unchanged, so exploration metrics should remain:
  - Ply 0 entropy: ~3.18 ✅
  - Early game entropy: ~3.998 ✅

## Validation Checklist

After retraining, verify:
- [ ] Normalized entropy median in [0.45, 0.65] (sharper policies)
- [ ] Training losses continue decreasing
- [ ] No training instabilities (NaNs, exploding gradients)
- [ ] Model performance improves vs previous Cycle 2 model

## Rollback Plan

If results are worse:
1. Revert config changes: `epochs=10`, `steps_per_epoch=50`
2. Consider Option B: Adjust exploration parameters and regenerate self-play

## Summary

**Approach**: Keep excellent self-play data, increase training
**Changes**: +50% epochs, +20% steps per epoch
**Expected**: Sharper policies while maintaining excellent exploration
**Status**: ✅ Ready to retrain

---

**Ready to proceed**: Run `make train CYCLE=2` to retrain with adjusted parameters.

