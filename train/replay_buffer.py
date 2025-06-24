import random
from typing import List, Tuple

import torch


class ReplayBuffer:
    """In-memory circular buffer used to store self-play examples."""

    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self.buffer: List[Tuple[torch.Tensor, torch.Tensor, int]] = []

    def add(self, game_data: List[Tuple[torch.Tensor, torch.Tensor, int]]) -> None:
        """
        Accepts a list of (state_tensor, pi_tensor, z_value) tuples from one self-play game.
        """
        self.buffer.extend(game_data)

        # Trim if over capacity (FIFO)
        overflow = len(self.buffer) - self.max_size
        if overflow > 0:
            self.buffer = self.buffer[overflow:]

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns a batch of (state, policy, value) as PyTorch tensors.
        """
        batch = random.sample(self.buffer, batch_size)
        states, policies, values = zip(*batch)

        return (
            torch.stack(states),  # Shape: [B, 3, 8, 8]
            torch.stack(policies),  # Shape: [B, 8, 8]
            torch.tensor(values, dtype=torch.long),  # Shape: [B]
        )

    def get_all_targets(self) -> List[int]:
        return [sample[2] for sample in self.buffer]

    @classmethod
    def load(cls, path: str) -> "ReplayBuffer":
        import pickle

        with open(path, "rb") as f:
            return pickle.load(f)

    def save(self, path: str) -> None:
        import pickle

        with open(path, "wb") as f:
            # noinspection PyTypeChecker
            pickle.dump(self, f)

    def __len__(self) -> int:
        return len(self.buffer)
