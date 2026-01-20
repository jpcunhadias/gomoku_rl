# Cycle 1 Exploration Analysis & Proposed Changes

## Problem Summary

**Critical Issue**: Exploration is collapsing at ply 0, with 61.3% of first moves having entropy < 1.0.

### Key Findings

#### Ply 0 Metrics (First Move - Root MCTS)
- **Entropy MCTS**: median=0.78, mean=0.91 (Target: >2.0)
- **Distribution**: 
  - 32.7% have entropy < 0.5 (very low)
  - 28.7% have entropy 0.5-1.0 (low)
  - Only 21.3% have entropy > 1.5 (acceptable)
- **Current Config**:
  - tau (temperature) = 0.25
  - dirichlet_epsilon = 0.02 (way too low!)
  - dirichlet_alpha = 0.04

#### Game-Level Stats
- 53.3% of games have early entropy < 1.5
- Median early game entropy: 1.47 (should be >2.0)

#### Plies 1-2
- Ply 1: entropy=3.28 ✓ (good, tau=0.12)
- Ply 2: entropy=1.62 (borderline, tau=0.06)
- Ply 3+: entropy=0.0 (greedy, tau=0, expected)

## Root Cause Analysis

The config file shows a mismatch between intended parameters and what's being recorded:

**Config (`phaseC_c1.py`):**
```python
tau_early_plies={0: 0.55, 1: 0.35, 2: 0.25}  # Higher temps
dirichlet_epsilon_root=0.40                   # High root noise
```

**Actual values in selfplay data:**
```
tau = 0.25           # ❌ Using tau_early, not tau_early_plies[0]
dirichlet_eps = 0.02 # ❌ Using dirichlet_epsilon, not dirichlet_epsilon_root
```

### Issue 1: Tau Not Applied at Ply 0 ⚠️ **BUG FOUND**
Looking at `self_play.py` line 122:
```python
current_player.set_temperature(self._tau_for_move(move_number))
```

The `_tau_for_move()` method (line 356-375) checks `tau_early_plies` mapping.

**Root cause**: During cycle 1, the config was serialized to JSON then loaded, converting integer keys {0: 0.55, 1: 0.35, 2: 0.25} to string keys {"0": 0.55, "1": 0.35, "2": 0.25}. When `_tau_for_move()` checks `if move_number in tau_early_plies`, it compares integer `move_number=0` against string keys, which fails.

**Result**: Falls back to `tau_early=0.15`, but the logged value shows 0.25 (from previous config?). Either way, the intended 0.55 was NOT used.

**Fix**: The current `phaseC_c1.py` has correct integer keys. We need to ensure config loading doesn't convert them to strings, OR fix `_tau_for_move()` to handle both int and str keys.

### Issue 2: Root Dirichlet Epsilon Logging Incorrect ⚠️ **LOGGING BUG**
Looking at `player.py` line 123-127:
```python
epsilon = (
    self.dirichlet_epsilon_root
    if self.move_number == 0
    else self.dirichlet_epsilon
)
```

The code correctly uses `dirichlet_epsilon_root=0.40` at ply 0, BUT the logged value is 0.02.

**Root cause**: The logging (line 206 in `self_play.py`) always logs:
```python
dirichlet_eps=getattr(current_player, "dirichlet_epsilon", 0.25)
```

This logs the non-root epsilon value, not the actual epsilon used! The actual application is correct (epsilon_root IS used), but our analysis was misled by wrong logging.

**Impact**: The actual epsilon at ply 0 was likely 0.40 (intended), not 0.02 (logged). However, given the low entropy, even 0.40 was insufficient.

**Fix**: Changed logging to capture actual epsilon used based on move_number and root_noise flag.

### Issue 3: Dirichlet Epsilon Too Low
Even if the logging is wrong, config shows:
```python
dirichlet_epsilon=0.05      # Non-root noise (too low)
dirichlet_epsilon_root=0.40 # Root noise (good, if applied)
```

The value 0.05 is too conservative for plies > 0.

## Proposed Changes

### Change 0: Fix tau_early_plies Bug 🔧 **CRITICAL**
**Bug**: `_tau_for_move()` fails when config is serialized/deserialized (int keys → str keys)
**Fix**: Add type conversion in `_tau_for_move()`:
```python
if tau_early_plies and (move_number in tau_early_plies or str(move_number) in tau_early_plies):
    return float(tau_early_plies.get(move_number, tau_early_plies.get(str(move_number))))
```

### Change 1: Increase Tau at Ply 0 ✓
**Current**: tau_early_plies={0: 0.55, 1: 0.35, 2: 0.25} (was NOT applied due to bug!)
**Actual used in cycle 1**: 0.25 (fallback)
**Proposed**: tau_early_plies={0: 0.85, 1: 0.50, 2: 0.30}

**Rationale**: Higher tau at ply 0 increases stochasticity in move selection, forcing exploration of diverse openings.

### Change 2: Increase Root Dirichlet Noise ✓
**Current**: dirichlet_epsilon_root=0.40
**Proposed**: dirichlet_epsilon_root=0.55

**Rationale**: More noise injection at root means priors are heavily mixed with uniform noise, preventing premature convergence to a few preferred openings.

### Change 3: Increase Non-Root Dirichlet Noise
**Current**: dirichlet_epsilon=0.05
**Proposed**: dirichlet_epsilon=0.12

**Rationale**: Plies 1-2 also need more exploration support. Current value is too conservative.

### Change 4: Adjust Dirichlet Alpha Range (Optional)
**Current**: 
```python
dirichlet_alpha_min=0.01
dirichlet_alpha_max=0.08
dirichlet_concentration=6.0
```
**Proposed**:
```python
dirichlet_alpha_min=0.015
dirichlet_alpha_max=0.10
dirichlet_concentration=8.0
```

**Rationale**: Slightly wider alpha range allows for more varied noise characteristics. Higher concentration (8.0) with more legal moves at ply 0 (~64) gives alpha ≈ 8/64 = 0.125 (capped at 0.10), which is stronger than current 6/64 = 0.094 (capped at 0.08).

### Change 5: Increase C_PUCT at Root (High Priority)
**Current**: 
```python
c_puct_schedule={"enabled": True, "c0": 0.3, "lambda_": 0.7, "c_min": 1.0}
c_puct=1.5  # base
```

At depth 0: c_puct_eff = 1.5 + 0.3 * exp(-0.7 * 0) = 1.5 + 0.3 = 1.8

**Proposed**:
```python
c_puct_schedule={"enabled": True, "c0": 0.8, "lambda_": 0.5, "c_min": 1.0}
```

At depth 0: c_puct_eff = 1.5 + 0.8 * exp(-0.5 * 0) = 1.5 + 0.8 = 2.3

**Rationale**: Higher c_puct at root increases exploration bonus in PUCT formula, encouraging MCTS to try less-visited nodes even if they have lower Q values. The slower decay (lambda=0.5) maintains exploration bonus deeper in the tree.

### Change 6: Reduce Early Game Simulation Budget (Optional)
**Current**: sim_budget={"early": 750, "mid": 200, "late": 120}
**Proposed**: sim_budget={"early": 600, "mid": 200, "late": 120}

**Rationale**: With 750 sims at ply 0, the policy converges too strongly despite noise. Fewer sims + higher tau + more noise = better exploration. This is optional but can help if other changes aren't sufficient.

## Expected Impact

With these changes:

1. **Ply 0 entropy**: Should increase from ~0.78 to >2.0
   - Higher tau (0.85 vs 0.25) provides 3.4x more stochasticity
   - Higher epsilon_root (0.55 vs 0.40) injects 37.5% more noise
   - Higher c_puct (2.3 vs 1.8) increases exploration by 28%

2. **Early game diversity**: 
   - Currently 53.3% games have early entropy < 1.5
   - Target: <20% games with early entropy < 1.5

3. **Opening variety**:
   - More varied first moves
   - Better coverage of opening space
   - Reduced tendency to always play center or same patterns

## Trade-offs

- **Training signal quality**: Higher exploration may introduce more "noise" in early training positions, but this is beneficial for learning robust policies
- **Computation time**: If we reduce sim_budget (optional change 6), we save ~20% compute in early game
- **Convergence speed**: More exploration may slow convergence slightly, but prevents premature convergence to suboptimal openings

## Implementation Priority

**High Priority (Must Have for Cycle 2)**:
1. ✅ Change 1: Increase tau at ply 0 to 0.85
2. ✅ Change 2: Increase dirichlet_epsilon_root to 0.55  
3. ✅ Change 5: Increase c_puct_schedule c0 to 0.8, lambda to 0.5

**Medium Priority (Should Have)**:
4. ✅ Change 3: Increase dirichlet_epsilon to 0.12

**Low Priority (Nice to Have)**:
5. Change 4: Adjust dirichlet alpha range
6. Change 6: Reduce early sim budget to 600

## Validation

After implementing changes, check:
- [ ] Ply 0 entropy median > 2.0
- [ ] Ply 0 entropy mean > 2.2
- [ ] < 5% of ply 0 samples with entropy < 1.0
- [ ] < 20% of games with early entropy < 1.5
- [ ] Game-level early entropy median > 2.0

## Next Steps

1. Update `configs/phaseC_c1.py` with proposed changes
2. Run cycle 2 selfplay with new config
3. Analyze cycle 2 results using same metrics
4. Compare ply 0 entropy distributions
5. If still low, consider more aggressive changes (tau=1.0, eps_root=0.70)
