#!/usr/bin/env python3
"""
Check normalized entropy of MCTS target policies in training data.
This tells us if the training data itself is too uniform.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import math
from train.replay_buffer import ReplayBuffer

def entropy(p):
    eps = 1e-12
    q = np.clip(p, eps, 1.0)
    return float(-(q * np.log(q)).sum())

# Load buffer
buffer = ReplayBuffer.load('checkpoints/buffers/replay_c1_cycle2.pkl')
print(f'Buffer size: {len(buffer)} samples')
print()

# Sample a large batch to get good statistics
batch_size = min(2048, len(buffer))
states, target_pi, values = buffer.sample(batch_size)
target_pi_np = target_pi.cpu().numpy()

# Calculate normalized entropy for MCTS policies (training targets)
h_norms_mcts = []
for b in range(len(target_pi_np)):
    pi = target_pi_np[b]
    legal_mask = pi > 1e-12
    m = int(legal_mask.sum())
    if m > 1:
        pi_legal = pi[legal_mask]
        h = entropy(pi_legal)
        h_norm = h / math.log(m)
        h_norms_mcts.append(h_norm)

h_norms_mcts = np.array(h_norms_mcts)
print('=== MCTS TARGET POLICY ENTROPY IN TRAINING DATA ===')
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
print('=== DIAGNOSIS ===')
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
print('=== COMPARISON WITH MODEL ===')
print('From debug report:')
print('  Model normalized entropy: mean=0.929, median=0.970')
print('  KL divergence: mean=0.012 (very low - model matches targets well)')
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

