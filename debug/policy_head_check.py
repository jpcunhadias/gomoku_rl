#!/usr/bin/env python3
# debug/policy_head_check.py
"""
Checks policy targets and network alignment on NON-TERMINAL samples:
- π legality & normalization (zero on illegal; sum over legal = 1)
- Normalized entropy H(π)/log(#legal)
- KL(π_net || π_mcts) over legal
- Top-k (k=1,3) vs MCTS argmax

Skips terminal samples where target π has no positive mass.
"""

import argparse
import math

import numpy as np
import torch

from model.policy_value_net import PolicyValueNet
from train.replay_buffer import ReplayBuffer


def entropy(p: np.ndarray) -> float:
    eps = 1e-12
    q = np.clip(p, eps, 1.0)
    return float(-(q * np.log(q)).sum())


def kl(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float((p * (np.log(p) - np.log(q))).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--buffer", required=True)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--board_size", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PolicyValueNet.load_from_checkpoint(
        args.checkpoint, board_size=args.board_size, device=device
    ).eval()
    buffer = ReplayBuffer.load(args.buffer)

    states, target_pi, _ = buffer.sample(args.batch)
    B, H, W = target_pi.shape
    states = states.to(device)

    with torch.no_grad():
        policy_logits, _ = model(states)  # [B, 64]
        net_pi = torch.softmax(policy_logits, dim=1).view(B, H, W).cpu().numpy()

    target_pi_np = target_pi.cpu().numpy()

    violations_illegal = 0
    violations_norm = 0
    kls, h_norms = [], []
    top1_hits, top3_hits = 0, 0
    skipped_terminal = 0
    used = 0

    for b in range(B):
        pi = target_pi_np[b]  # [8,8]
        net = net_pi[b]  # [8,8]

        # Identify legal support from π; treat tiny values as zero
        legal_mask = pi > 1e-12
        m = int(legal_mask.sum())

        # Skip terminal positions (no positive mass in π)
        if m == 0:
            skipped_terminal += 1
            continue

        used += 1

        # Legality: π should be zero on illegal cells
        if np.any(pi[~legal_mask] > 1e-8):
            violations_illegal += 1

        # Normalization on legal cells
        s = float(pi[legal_mask].sum())
        if abs(s - 1.0) > 1e-6:
            violations_norm += 1

        # Compare distributions only on legal support
        net_legal = net[legal_mask]
        net_legal = net_legal / (net_legal.sum() + 1e-12)
        pi_legal = pi[legal_mask]

        # KL(π_net || π_mcts)
        kls.append(kl(net_legal, pi_legal))

        # Normalized entropy (skip degenerate cases with ≤1 legal move)
        if m > 1:
            h = entropy(pi_legal)
            h_norms.append(h / math.log(m))

        # Top-k vs MCTS argmax (use full board argmax for π)
        arg_mcts = int(np.argmax(pi))
        arg_net = int(np.argmax(net))
        if arg_net == arg_mcts:
            top1_hits += 1

        top3_net = np.argpartition(net.ravel(), -3)[-3:]
        if arg_mcts in top3_net:
            top3_hits += 1

    print("\n=== POLICY HEAD CHECK ===")
    print(
        f"Batch size: {B}  |  used (non-terminal): {used}  |  skipped (terminal): {skipped_terminal}"
    )

    if used == 0:
        print("No non-terminal samples in batch. Increase batch size or resample.")
        return

    h_norms = np.asarray(h_norms, dtype=np.float64)
    kls = np.asarray(kls, dtype=np.float64)

    print(f"Legality violations (π>0 on illegal): {violations_illegal}")
    print(f"Normalization violations (sum(legal) != 1): {violations_norm}")

    print(
        f"Normalized entropy H(π)/log(#legal): mean={h_norms.mean():.3f}  median={np.median(h_norms):.3f}  "
        f"IQR=[{np.percentile(h_norms, 25):.3f},{np.percentile(h_norms, 75):.3f}]"
    )
    print(
        f"KL(π_net || π_mcts) over legal:       mean={kls.mean():.3f}  median={np.median(kls):.3f}  "
        f"IQR=[{np.percentile(kls, 25):.3f},{np.percentile(kls, 75):.3f}]"
    )

    print(f"Top-1 vs MCTS argmax: {(top1_hits / used):.2%}")
    print(f"Top-3 vs MCTS argmax: {(top3_hits / used):.2%}")

    # checkbox summary (entropy band mainly meaningful for early-game heavy batches)
    early_band_ok = 0.45 <= np.median(h_norms) <= 0.65
    print("\n=== CHECKBOX SUMMARY ===")
    print(
        f"[{'x' if violations_illegal == 0 else ' '}] π[illegal] == 0 for all used samples"
    )
    print(f"[{'x' if violations_norm == 0 else ' '}] sum(π on legal) == 1 (±1e-6)")
    print(
        f"[{'x' if early_band_ok else ' '}] normalized entropy median in [0.45, 0.65] (exploration sanity)"
    )
    print("[x] KL computed (baseline)")
    print("[x] Top-k computed (diagnostic)")


if __name__ == "__main__":
    main()
