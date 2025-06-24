import random
from typing import Tuple, Protocol, Any, Dict

import numpy as np

from game.gomoku import GomokuBoard
from mcts.mcts import MCTS


class Player(Protocol):
    """Protocol for player implementations."""

    def get_action(self, board: GomokuBoard) -> Tuple[int, int]:
        """Return the action ``(row, col)`` chosen on ``board``."""
        ...


class RandomPlayer:
    """Player that selects moves uniformly at random."""

    def __init__(self, player_id: int) -> None:
        self.player_id = player_id

    def __repr__(self) -> str:
        return f"RandomPlayer({self.player_id})"

    def get_action(self, board: GomokuBoard) -> Tuple[int, int]:
        """Return a randomly chosen legal move."""
        legal_moves = board.get_legal_moves()
        return random.choice(legal_moves)


class HumanPlayer:
    """Player that queries the user for a move via the console."""

    def __init__(self, player_id: int) -> None:
        self.player_id = player_id

    def get_action(self, board: GomokuBoard) -> Tuple[int, int]:
        """Prompt the user for a legal move."""
        while True:
            try:
                move_str = input(
                    f"Player {self.player_id}, enter your move as 'row,col': "
                )
                row, col = map(int, move_str.strip().split(","))
                if board.is_legal_move(row, col):
                    return (row, col)
                else:
                    print("Illegal move. Try again.")
            except Exception:
                print("Invalid input. Format must be: row,col (e.g. 7,7)")


class MCTSPlayer:
    """Player that selects moves using Monte-Carlo Tree Search."""

    def __init__(
        self,
        mcts: MCTS,
        temperature: float = 1e-3,
        add_dirichlet_noise: bool = False,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        name: str = "MCTSPlayer",
    ) -> None:
        self.mcts = mcts
        self.temperature = temperature
        self.add_dirichlet_noise = add_dirichlet_noise
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.move_number = 0  # Track move number internally
        self.name = name

    def __repr__(self) -> str:
        return f"{self.name}(temperature={self.temperature})"

    def set_temperature(self, temp: float) -> None:
        """Set the sampling temperature used when selecting moves."""
        self.temperature = temp

    def reset(self) -> None:
        """Reset internal move counter before a new game."""
        self.move_number = 0

    def get_action(self, board: GomokuBoard) -> Any:
        """Return a move selected by MCTS."""
        action_probs = self.mcts.get_action_probs(board, temp=self.temperature)

        if not action_probs:
            # Fallback: pick random move if MCTS failed
            print("[WARNING] MCTS returned no moves. Picking random legal move.")
            legal_moves = board.get_legal_moves()
            return random.choice(legal_moves)

        # === Add Dirichlet noise only on first move if enabled ===
        if self.add_dirichlet_noise and self.move_number == 0:
            action_probs = self._add_dirichlet_noise(action_probs)

        actions, probs = zip(*action_probs.items())

        if self.temperature <= 1e-3:
            # Select the move with the highest probability when deterministic
            selected_action = max(action_probs.items(), key=lambda x: x[1])[0]
        else:
            selected_action = random.choices(actions, weights=probs, k=1)[0]

        self.move_number += 1
        return selected_action

    def _add_dirichlet_noise(self, action_probs: Dict[Any, float]) -> Dict[Any, float]:
        """Inject Dirichlet noise into action probabilities."""
        actions = list(action_probs.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
        noisy_probs: Dict[Any, float] = {}
        for a, n in zip(actions, noise):
            noisy_probs[a] = (1 - self.dirichlet_epsilon) * action_probs[
                a
            ] + self.dirichlet_epsilon * n
        return noisy_probs
