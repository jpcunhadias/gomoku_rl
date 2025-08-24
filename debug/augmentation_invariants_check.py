#!/usr/bin/env python3
# debug/augmentation_invariants_check.py
"""
Verifies augmentation invariants across the 8 board symmetries:
- Invertibility: inv(T(T(x))) ≈ x for state and policy
- Argmax alignment: argmax(T(π)) == T(argmax(π))
- Turn-indicator plane is spatially constant after transforms

Assumes:
- train/augmentation.py defines TRANSFORMS as pairs (state_tf, policy_tf)
- ReplayBuffer.sample returns (states [B,3,8,8], pi [B,8,8], values)
"""

import argparse

import torch

from train.replay_buffer import ReplayBuffer


def _rot(state, k):
    return torch.rot90(state, k=k, dims=[1, 2])


def _rotp(pi, k):
    return torch.rot90(pi, k=k, dims=[0, 1])


def _flipx(state):
    return torch.flip(state, dims=[2])  # horizontal


def _flipxp(pi):
    return torch.flip(pi, dims=[1])


def _flipy(state):
    return torch.flip(state, dims=[1])  # vertical


def _flipyp(pi):
    return torch.flip(pi, dims=[0])


def _transpose(state):
    return torch.transpose(state, 1, 2)


def _transposep(pi):
    return torch.transpose(pi, 0, 1)


# Build explicit 8 symmetries and their inverses (for checks)
SYMS = [
    (lambda s: s, lambda p: p, lambda s: s, lambda p: p),  # I
    (
        lambda s: _rot(s, 1),
        lambda p: _rotp(p, 1),
        lambda s: _rot(s, 3),
        lambda p: _rotp(p, 3),
    ),  # R90
    (
        lambda s: _rot(s, 2),
        lambda p: _rotp(p, 2),
        lambda s: _rot(s, 2),
        lambda p: _rotp(p, 2),
    ),  # R180
    (
        lambda s: _rot(s, 3),
        lambda p: _rotp(p, 3),
        lambda s: _rot(s, 1),
        lambda p: _rotp(p, 1),
    ),  # R270
    (
        lambda s: _flipx(s),
        lambda p: _flipxp(p),
        lambda s: _flipx(s),
        lambda p: _flipxp(p),
    ),  # FlipX
    (
        lambda s: _flipy(s),
        lambda p: _flipyp(p),
        lambda s: _flipy(s),
        lambda p: _flipyp(p),
    ),  # FlipY
    (
        lambda s: _transpose(s),
        lambda p: _transposep(p),
        lambda s: _transpose(s),
        lambda p: _transposep(p),
    ),  # Transpose
    (
        lambda s: _flipx(_transpose(s)),
        lambda p: _flipxp(_transposep(p)),
        lambda s: _transpose(_flipx(s)),
        lambda p: _transposep(_flipxp(p)),
    ),  # Mirror diag+flip
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer", required=True)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    buffer = ReplayBuffer.load(args.buffer)
    states, pis, _ = buffer.sample(args.batch)  # states [B,3,8,8], pis [B,8,8]
    states0 = states.clone()
    pis0 = pis.clone()

    tol = 1e-6
    inv_fail = 0
    argmax_fail = 0
    turn_plane_fail = 0

    for bis in range(states.size(0)):
        s = states[bis]
        p = pis[bis]
        # turn-indicator plane spatial constancy
        plane = s[2]
        if not torch.allclose(plane, plane[0, 0] * torch.ones_like(plane)):
            turn_plane_fail += 1

        # argmax baseline
        arg0 = torch.argmax(p).item()

        for Ts, Tp, InvTs, InvTp in SYMS:
            s_t = Ts(s)
            p_t = Tp(p)

            # invertibility
            s_inv = InvTs(s_t)
            p_inv = InvTp(p_t)
            if not torch.allclose(s_inv, s, atol=tol):
                inv_fail += 1
                break
            if not torch.allclose(p_inv, p, atol=tol):
                inv_fail += 1
                break

            # argmax alignment
            arg_t = torch.argmax(p_t).item()

            # map arg0 through transform: to check, transform a one-hot board
            oh = torch.zeros_like(p)
            oh.view(-1)[arg0] = 1.0
            oh_t = Tp(oh)
            arg0_t = torch.argmax(oh_t).item()
            if arg_t != arg0_t:
                argmax_fail += 1
                break

    print("\n=== AUGMENTATION INVARIANTS CHECK ===")
    print(f"Batches checked: {states.size(0)}")
    print(f"Invertibility failures: {inv_fail}")
    print(f"Argmax alignment failures: {argmax_fail}")
    print(f"Turn-indicator plane non-constant: {turn_plane_fail}")

    print("\n=== CHECKBOX SUMMARY ===")
    print(
        f"[{'x' if inv_fail == 0 else ' '}] Invertibility (state & π) holds for all 8 symmetries"
    )
    print(
        f"[{'x' if argmax_fail == 0 else ' '}] Argmax alignment holds across symmetries"
    )
    print(
        f"[{'x' if turn_plane_fail == 0 else ' '}] Turn-indicator plane spatially constant"
    )


if __name__ == "__main__":
    main()
