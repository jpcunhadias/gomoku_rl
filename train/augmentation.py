"""
augmentation.py

Board and policy tensor augmentation utilities for self-play data enhancement.
Applies random flips, rotations, or mirrors to both the board state tensor and
its corresponding policy tensor.

State tensor shape: ``[3, 8, 8]``
Policy tensor shape: ``[8, 8]``
Channels:
    0 - Current player's stones
    1 - Opponent's stones
    2 - Turn indicator
"""

import random

import torch

# ---------------------------------------------------------------------------
# Transformation definitions
# Each entry is a pair of callables to transform the state and policy tensors
# respectively.  The same operation is used for both tensors to preserve the
# relationship between board state and policy targets.
# ---------------------------------------------------------------------------
TRANSFORMS: list[tuple] = [
    (
        lambda s: torch.flip(s, dims=[2]),  # Horizontal flip
        lambda p: torch.flip(p, dims=[1]),
    ),
    (
        lambda s: torch.flip(s, dims=[1]),  # Vertical flip
        lambda p: torch.flip(p, dims=[0]),
    ),
    (
        lambda s: torch.rot90(s, k=1, dims=[1, 2]),  # 90° rotation
        lambda p: torch.rot90(p, k=1, dims=[0, 1]),
    ),
    (
        lambda s: torch.rot90(s, k=2, dims=[1, 2]),  # 180° rotation
        lambda p: torch.rot90(p, k=2, dims=[0, 1]),
    ),
    (
        lambda s: torch.rot90(s, k=3, dims=[1, 2]),  # 270° rotation
        lambda p: torch.rot90(p, k=3, dims=[0, 1]),
    ),
    (
        lambda s: torch.transpose(s, 1, 2),  # Mirror (transpose)
        lambda p: torch.transpose(p, 0, 1),
    ),
]


def apply_random_transform(
    state: torch.Tensor, policy: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly apply a single transform to both ``state`` and ``policy``.

    With 50% probability no transformation is applied.

    Args:
        state (torch.Tensor): State tensor of shape ``[3, 8, 8]``.
        policy (torch.Tensor): Policy tensor of shape ``[8, 8]``.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Augmented (state, policy) tensors.
    """

    assert state.shape == (3, 8, 8), f"Expected state shape [3,8,8], got {state.shape}"
    assert policy.shape == (8, 8), f"Expected policy shape [8,8], got {policy.shape}"

    # 50% chance to apply augmentation
    if random.random() > 0.5:
        return state, policy

    state_tf, policy_tf = random.choice(TRANSFORMS)
    return state_tf(state), policy_tf(policy)


def augment_board_state(board_tensor: torch.Tensor) -> torch.Tensor:
    """Apply random augmentation to the board tensor with 50% probability."""

    augmented, _ = apply_random_transform(board_tensor, torch.zeros(8, 8))
    return augmented


def augment_data(
    data: list[tuple[torch.Tensor, torch.Tensor, float]],
) -> list[tuple[torch.Tensor, torch.Tensor, float]]:
    """Apply a random transform to both state and policy for each sample."""

    augmented = []
    for state, policy, value in data:
        aug_state, aug_policy = apply_random_transform(state, policy)
        augmented.append((aug_state, aug_policy, value))
    return augmented
