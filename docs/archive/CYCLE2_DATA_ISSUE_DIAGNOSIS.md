# Cycle 2 Data Issue Diagnosis

**Date**: After investigation revealed training data problem
**Status**: ✅ Root cause identified, solution ready

## Problem Confirmed

Investigation using `debug/check_mcts_target_entropy.py` revealed:

### MCTS Target Policies in Training Data
- **Median normalized entropy: 0.973** (target: 0.45-0.65) ⚠️
- **Mean normalized entropy: 0.928**
- **Distribution**: 82.1% of samples have entropy >0.80 (too uniform!)

### Model Policies
- **Median normalized entropy: 0.970** (matches targets!)
- **KL divergence: 0.012** (very low - model correctly learns from targets)

## Root Cause

**The training data itself is the problem!**

1. Cycle 2 exploration parameters were set very high (tau=0.85, dirichlet_eps=0.55)
2. This generated excellent raw exploration metrics (entropy 3.18 at ply 0) ✅
3. BUT it also generated uniform MCTS policies (normalized entropy 0.973) ⚠️
4. Model correctly learned these uniform policies (KL=0.012)
5. Result: Model has uniform policies because it learned from uniform data

## The Confusion

We initially thought:
- ✅ Self-play data is excellent (raw entropy 3.18)
- ❌ Model needs more training to learn sharper policies

**Reality:**
- ✅ Self-play has good raw exploration
- ❌ But MCTS policies are too uniform (normalized entropy 0.973)
- ✅ Model correctly learns uniform policies
- ❌ Need to regenerate self-play with reduced exploration

## Solution: Regenerate Self-Play with Reduced Exploration

### Changes Applied to `phaseC_c2.py` (v3)

| Parameter | Before (v2) | After (v3) | Change | Rationale |
|-----------|-------------|------------|--------|-----------|
| `tau_early_plies[0]` | 0.85 | **0.70** | -18% | Less stochastic, sharper policies |
| `tau_early_plies[1]` | 0.50 | **0.42** | -16% | Moderate reduction |
| `tau_early_plies[2]` | 0.30 | **0.28** | -7% | Slight reduction |
| `dirichlet_epsilon_root` | 0.55 | **0.45** | -18% | Less noise at root |
| `dirichlet_epsilon` | 0.12 | **0.08** | -33% | Less noise at non-root |
| `c_puct_schedule c0` | 0.8 | **0.6** | -25% | Less aggressive exploration |

### Training Parameters (Kept from v2)
- `epochs`: 15 (increased from 10)
- `steps_per_epoch`: 60 (increased from 50)

## Expected Impact

### MCTS Target Policies
- Normalized entropy should decrease from 0.973 → 0.45-0.65 range
- Policies will be sharper and more decisive
- Still maintain good exploration (raw entropy >2.0)

### Model Policies
- Will follow MCTS targets (low KL divergence)
- Normalized entropy should decrease toward 0.45-0.65
- Sharper, more decisive policies

### Exploration Metrics
- Raw entropy at ply 0 should remain >2.0 (still good exploration)
- Early game entropy should remain >2.0
- But normalized entropy will be in target range

## Next Steps

### 1. Clean Cycle 2 Artifacts

```bash
# Delete old self-play data and buffer
rm checkpoints/buffers/replay_c1_cycle2.pkl
rm checkpoints/selfplay/selfplay_c1_cycle2.jsonl
rm checkpoints/selfplay/c1_cycle2_summary.json
rm checkpoints/models/c1_cycle2_*.pth  # Already deleted
```

### 2. Regenerate Self-Play

```bash
make self-play CYCLE=2
```

This will:
- Use updated config with reduced exploration
- Seed from Cycle 1 buffer (if exists)
- Generate new self-play data with sharper MCTS policies

### 3. Validate New Data

```bash
# Check MCTS target entropy in new buffer
python debug/check_mcts_target_entropy.py

# Should show normalized entropy in 0.45-0.65 range
```

### 4. Train Model

```bash
make train CYCLE=2
```

### 5. Validate Results

```bash
make debug CYCLE=2
```

Should show:
- ✅ Normalized entropy in 0.45-0.65 range
- ✅ Sharper policies
- ✅ Good exploration metrics maintained

## Key Insight

**Normalized entropy measures policy sharpness, not exploration!**

- **Raw entropy** (3.18): Measures exploration diversity ✅
- **Normalized entropy** (0.973): Measures policy sharpness ❌

We achieved excellent exploration but at the cost of policy sharpness. The solution is to balance both:
- Good exploration (raw entropy >2.0)
- Sharp policies (normalized entropy 0.45-0.65)

## Files Modified

- ✅ `configs/phaseC_c2.py` - Reduced exploration parameters (v3)
- ✅ `debug/check_mcts_target_entropy.py` - Diagnostic script created

---

**Status**: Ready to regenerate self-play with corrected exploration parameters.

