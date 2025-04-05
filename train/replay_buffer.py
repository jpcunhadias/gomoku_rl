import random
from typing import List, Tuple
import torch


class ReplayBuffer:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer: List[Tuple[torch.Tensor, torch.Tensor, float]] = []

    def add(self, game_data: List[Tuple[torch.Tensor, torch.Tensor, float]]):
        """
        Accepts a list of (state_tensor, pi_tensor, z_value) tuples from one self-play game.
        """
        self.buffer.extend(game_data)

        # Trim if over capacity (FIFO)
        overflow = len(self.buffer) - self.max_size
        if overflow > 0:
            self.buffer = self.buffer[overflow:]

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns a batch of (state, policy, value) as PyTorch tensors.
        """
        batch = random.sample(self.buffer, batch_size)
        states, policies, values = zip(*batch)

        return (
            torch.stack(states),  # Shape: [B, 3, 15, 15]
            torch.stack(policies),  # Shape: [B, 15, 15]
            torch.tensor(values, dtype=torch.float32).unsqueeze(1),  # Shape: [B, 1]
        )

    def __len__(self):
        return len(self.buffer)
