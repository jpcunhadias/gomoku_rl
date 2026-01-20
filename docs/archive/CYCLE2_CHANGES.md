# Cycle 2 Changes Summary

## Executive Summary

**Problem**: Cycle 1 showed severe exploration collapse at ply 0, with 61.3% of first moves having entropy < 1.0 (target: >2.0).

**Root Causes**:
1. **Bug**: `tau_early_plies` dict not applied due to int/str key mismatch after JSON serialization
2. **Config**: Even intended values were too conservative (tau=0.55, epsilon_root=0.40)
3. **Logging**: `dirichlet_eps` logging showed wrong value (logged `dirichlet_epsilon` instead of actual epsilon used)

**Solutions**:
1. ✅ Fixed `_tau_for_move()` to handle both int and str keys
2. ✅ Fixed `dirichlet_eps` logging to show actual epsilon used
3. ✅ Created `phaseC_c2.py` with more aggressive exploration parameters

## Code Fixes Applied

### 1. Fix tau_early_plies Bug (train/self_play.py)

**Before:**
```python
if tau_early_plies and move_number in tau_early_plies:
    return float(tau_early_plies[move_number])
```

**After:**
```python
if tau_early_plies:
    # Handle both int and str keys (JSON serialization converts int keys to str)
    if move_number in tau_early_plies:
        return float(tau_early_plies[move_number])
    elif str(move_number) in tau_early_plies:
        return float(tau_early_plies[str(move_number)])
```

**Impact**: Now correctly applies per-ply tau values even after config serialization/deserialization.

### 2. Fix dirichlet_eps Logging (train/self_play.py)

**Before:**
```python
dirichlet_eps=getattr(current_player, "dirichlet_epsilon", 0.25),
```

**After:**
```python
eps_eff = 0.0
if getattr(current_player, "add_dirichlet_noise", False) and root_noise:
    alpha_eff = current_player.get_dirichlet_alpha(board)
    eps_eff = (
        current_player.dirichlet_epsilon_root
        if move_number == 0
        else current_player.dirichlet_epsilon
    )
...
dirichlet_eps=eps_eff,
```

**Impact**: Logs show actual epsilon used (root epsilon at ply 0, regular epsilon at other plies with noise).

## Configuration Changes (phaseC_c2.py)

| Parameter | Cycle 1 | Cycle 2 | Change | Rationale |
|-----------|---------|---------|--------|-----------|
| **tau_early_plies[0]** | 0.55 | **0.85** | +54% | Much more stochastic move selection at ply 0 |
| **tau_early_plies[1]** | 0.35 | **0.50** | +43% | Better exploration at ply 1 |
| **tau_early_plies[2]** | 0.25 | **0.30** | +20% | Maintain some exploration at ply 2 |
| **dirichlet_epsilon_root** | 0.40 | **0.55** | +38% | Stronger noise injection at root |
| **dirichlet_epsilon** | 0.05 | **0.12** | +140% | More exploration at non-root early plies |
| **c_puct_schedule.c0** | 0.3 | **0.8** | +167% | Much stronger exploration bonus |
| **c_puct_schedule.lambda_** | 0.7 | **0.5** | -29% | Slower decay, maintains exploration longer |
| **dirichlet_concentration** | 6.0 | **8.0** | +33% | Slightly stronger noise concentration |
| **dirichlet_alpha_max** | 0.08 | **0.10** | +25% | Allow stronger alpha in some cases |

### Effective c_puct at Root

**Cycle 1**: c_puct_eff = 1.5 + 0.3 × exp(-0.7 × 0) = **1.8**
**Cycle 2**: c_puct_eff = 1.5 + 0.8 × exp(-0.5 × 0) = **2.3** (+28%)

## Expected Results

### Ply 0 Entropy
- **Cycle 1**: median=0.78, mean=0.91
- **Cycle 2 Target**: median>2.0, mean>2.2
- **Expected Improvement**: ~2.5x increase

### Early Game Entropy
- **Cycle 1**: median=1.47, 53% of games < 1.5
- **Cycle 2 Target**: median>2.0, <20% of games < 1.5
- **Expected Improvement**: ~1.4x increase, 2.7x reduction in low-entropy games

### Opening Diversity
- More varied first moves (currently biased toward center)
- Better coverage of opening space
- Reduced repetition of same patterns

## Usage

To run cycle 2 with new config:

```bash
# Option 1: Use new config directly
python cli/train/train_loop_main.py --config configs.phaseC_c2

# Option 2: Load previous checkpoint and continue
python cli/train/train_loop_main.py --config configs.phaseC_c2 --load checkpoints/policy_value_net_best.pth
```

## Validation Checklist

After cycle 2 completes, verify:

- [ ] Ply 0 entropy median > 2.0
- [ ] Ply 0 entropy mean > 2.2
- [ ] < 5% of ply 0 samples with entropy < 1.0
- [ ] < 20% of games with early entropy < 1.5
- [ ] Game-level early entropy median > 2.0
- [ ] Check logged tau values match expected (0.85, 0.50, 0.30)
- [ ] Check logged dirichlet_eps at ply 0 is 0.55 (not 0.12)

## Analysis Script

```python
import json
import statistics

with open('checkpoints/selfplay/selfplay_c1_cycle2.jsonl', 'r') as f:
    samples = [json.loads(line) for line in f if 'type' not in json.loads(line) or json.loads(line)['type'] != 'game_summary']

ply0 = [s for s in samples if s['move_number'] == 0]
entropy_ply0 = [s['entropy_pi_mcts'] for s in ply0]

print(f"Ply 0 entropy: median={statistics.median(entropy_ply0):.3f}, mean={statistics.mean(entropy_ply0):.3f}")
print(f"Low entropy samples: {sum(1 for e in entropy_ply0 if e < 1.0)}/{len(entropy_ply0)} ({100*sum(1 for e in entropy_ply0 if e < 1.0)/len(entropy_ply0):.1f}%)")
```

## Rollback Plan

If cycle 2 shows worse results (e.g., training instability, model degradation):

1. Revert to `phaseC_c1.py` config
2. Try intermediate values:
   - tau_early_plies={0: 0.70, 1: 0.42, 2: 0.28}
   - dirichlet_epsilon_root=0.48
   - c_puct_schedule c0=0.55

## Next Steps

1. ✅ Code fixes applied
2. ✅ Config created (phaseC_c2.py)
3. ⏳ Run cycle 2 selfplay
4. ⏳ Analyze cycle 2 results
5. ⏳ Compare with cycle 1 metrics
6. ⏳ Iterate if needed
