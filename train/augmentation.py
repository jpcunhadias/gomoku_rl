"""
augmentation.py

Board state augmentation utilities for self-play data enhancement.
Applies random flips, rotations, or mirrors to the board state tensor.

Input tensor shape: [3, 8, 8]
Channels:
    0 - Current player's stones
    1 - Opponent's stones
    2 - Turn indicator
"""

import torch
import random


def augment_board_state(board_tensor: torch.Tensor) -> torch.Tensor:
    """
    Apply random augmentation to the board tensor with 50% probability.

    Args:
        board_tensor (torch.Tensor): A tensor of shape [3, 8, 8].

    Returns:
        torch.Tensor: Augmented tensor (or original if no augmentation applied).
    """
    # Validate input
    assert board_tensor.shape == (
        3,
        8,
        8,
    ), f"Expected shape [3,8,8], got {board_tensor.shape}"

    # 50% chance to apply augmentation
    if random.random() > 0.5:
        return board_tensor  # No augmentation, return as is

    # Choose a random transformation
    transformations = [
        lambda x: torch.flip(x, dims=[2]),  # Horizontal flip
        lambda x: torch.flip(x, dims=[1]),  # Vertical flip
        lambda x: torch.rot90(x, k=1, dims=[1, 2]),  # 90 degrees rotation
        lambda x: torch.rot90(x, k=2, dims=[1, 2]),  # 180 degrees rotation
        lambda x: torch.rot90(x, k=3, dims=[1, 2]),  # 270 degrees rotation
        lambda x: torch.transpose(x, 1, 2),  # Mirror (transpose)
    ]

    transform = random.choice(transformations)
    augmented = transform(board_tensor)

    return augmented


def augment_data(data):
    """Apply ``augment_board_state`` to each sample in the list.

    Args:
        data (list): Sequence of ``(state, policy, value)`` tuples.

    Returns:
        list: List with augmented state tensors.
    """
    return [
        (augment_board_state(state), policy, value) for state, policy, value in data
    ]
