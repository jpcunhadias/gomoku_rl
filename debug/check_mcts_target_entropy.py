#!/usr/bin/env python3
"""
Check normalized entropy of MCTS target policies in training data.
This tells us if the training data itself is too uniform.

Reports both a pooled aggregate (kept for backward-compat with the 0.45-0.65 target used
throughout docs/CHANGELOG.md) and a per-ply breakdown. The pooled number can be misleading:
plies where tau_early_plies stops applying (>= tau_cutoff_plies) collapse to near-deterministic
regardless of exploration settings once the model has real signal, and even within the tau
window later plies (visit counts already concentrated by the search itself) respond much less
to tau than ply 0 does. Pooling all of them into one median can hide this -- see
docs/current/SWEEP_TAU_DIRICHLET.md for the investigation that found this.

Ply is recovered from the state tensor itself (stone count = ply number; channels 0/1 are the
current player's and opponent's stones), not stored separately in the replay buffer.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse

import numpy as np
import math
from train.replay_buffer import ReplayBuffer


def entropy(p):
    eps = 1e-12
    q = np.clip(p, eps, 1.0)
    return float(-(q * np.log(q)).sum())


def print_stats(label, h_norms):
    h_norms = np.asarray(h_norms, dtype=np.float64)
    if len(h_norms) == 0:
        print(f'{label}: no samples with >1 legal move in target (all one-hot/terminal)')
        return None
    median = float(np.median(h_norms))
    print(
        f'{label}: n={len(h_norms):4d}  mean={h_norms.mean():.3f}  median={median:.3f}  '
        f'IQR=[{np.percentile(h_norms, 25):.3f}, {np.percentile(h_norms, 75):.3f}]  '
        f'min={h_norms.min():.3f} max={h_norms.max():.3f}'
    )
    return median


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--buffer",
    default="checkpoints/buffers/replay_c1_cycle2.pkl",
    help="Path to the replay buffer .pkl to check",
)
parser.add_argument(
    "--tau_cutoff_plies",
    type=int,
    default=3,
    help="Plies with tau>0 in self-play (matches config's tau_cutoff_plies); "
    "only these get their own row, the rest are pooled as 'ply >= cutoff'",
)
args = parser.parse_args()

# Load buffer
buffer = ReplayBuffer.load(args.buffer)
print(f'Buffer size: {len(buffer)} samples')
print()

# Sample a large batch to get good statistics
# Early-ply (0/1/2) samples are inherently rare -- roughly one per game, so capping at
# 2048 out of a much larger buffer was throwing away most of the already-small pool of
# ply-eligible samples for no benefit (this is pure CPU/numpy work, not model inference,
# so there's no cost to using the whole buffer).
batch_size = len(buffer)
states, target_pi, values = buffer.sample(batch_size)
target_pi_np = target_pi.cpu().numpy()
states_np = states.cpu().numpy()
# ply = total stones on board = current-player plane + opponent plane, summed
ply_np = states_np[:, 0, :, :].sum(axis=(1, 2)) + states_np[:, 1, :, :].sum(axis=(1, 2))
ply_np = ply_np.round().astype(int)

# Calculate normalized entropy for MCTS policies (training targets), keyed by ply
h_by_ply = {}
h_all = []
for b in range(len(target_pi_np)):
    pi = target_pi_np[b]
    legal_mask = pi > 1e-12
    m = int(legal_mask.sum())
    if m > 1:
        pi_legal = pi[legal_mask]
        h_norm = entropy(pi_legal) / math.log(m)
        h_all.append(h_norm)
        h_by_ply.setdefault(int(ply_np[b]), []).append(h_norm)

print('=== MCTS TARGET POLICY ENTROPY, PER PLY ===')
for ply in sorted(k for k in h_by_ply if k < args.tau_cutoff_plies):
    print_stats(f'  ply={ply}', h_by_ply[ply])
cutoff_pooled = [h for p, hs in h_by_ply.items() if p >= args.tau_cutoff_plies for h in hs]
print_stats(f'  ply>={args.tau_cutoff_plies} (tau=0, pooled)', cutoff_pooled)

print()
print('=== MCTS TARGET POLICY ENTROPY, POOLED ACROSS ALL PLIES (legacy aggregate) ===')
print('NOTE: mixes a ply where tau clearly matters (0) with plies where it barely does once')
print('the model has real signal (see per-ply breakdown above) -- treat this number with')
print('caution, not as the primary signal.')
h_norms_mcts = np.array(h_all)
print(f'Samples analyzed: {len(h_norms_mcts)}')
print(f'Mean normalized entropy: {h_norms_mcts.mean():.3f}')
print(f'Median normalized entropy: {np.median(h_norms_mcts):.3f}')
print(f'IQR: [{np.percentile(h_norms_mcts, 25):.3f}, {np.percentile(h_norms_mcts, 75):.3f}]')
print(f'Min: {h_norms_mcts.min():.3f}, Max: {h_norms_mcts.max():.3f}')
print()
print('Distribution:')
bins = [(0, 0.3), (0.3, 0.45), (0.45, 0.65), (0.65, 0.8), (0.8, 1.0)]
for low, high in bins:
    count = np.sum((h_norms_mcts >= low) & (h_norms_mcts < high))
    pct = 100 * count / len(h_norms_mcts)
    bar = '█' * int(pct / 5)
    print(f'  {low:.2f}-{high:.2f}: {count:4d} ({pct:5.1f}%) {bar}')

print()
print('=== DIAGNOSIS (pooled aggregate, kept for backward-compat) ===')
median_entropy = np.median(h_norms_mcts)
if median_entropy > 0.65:
    print(f'⚠️  PROBLEM FOUND: MCTS target policies are too uniform!')
    print(f'   Median normalized entropy: {median_entropy:.3f} (target: 0.45-0.65)')
    print(f'   The model is correctly learning from uniform data.')
    print(f'   Solution: Reduce exploration parameters in self-play config.')
elif median_entropy < 0.45:
    print(f'⚠️  MCTS target policies are too sharp (low entropy)')
    print(f'   Median normalized entropy: {median_entropy:.3f} (target: 0.45-0.65)')
    print(f'   Solution: Increase exploration parameters.')
else:
    print(f'✅ MCTS target policies are in good range')
    print(f'   Median normalized entropy: {median_entropy:.3f} (target: 0.45-0.65)')
    print(f'   If model entropy is still high, training issue.')

print()
if median_entropy > 0.65:
    print('CONCLUSION: Training data is the problem!')
    print('  - MCTS targets are too uniform (high entropy)')
    print('  - Model correctly learns uniform policies (low KL)')
    print('  - Need to regenerate self-play with reduced exploration')
else:
    print('CONCLUSION: Training issue, not data issue')
    print('  - MCTS targets are in good range')
    print('  - Model should learn sharper policies')
    print('  - May need training adjustments')
