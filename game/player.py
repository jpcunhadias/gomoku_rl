import random
from typing import Any, Dict, Protocol, Tuple, Union

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
        dirichlet_alpha: Union[float, str] = "auto",
        dirichlet_epsilon: float = 0.25,
        name: str = "MCTSPlayer",
        dirichlet_alpha_min: float = 0.02,
        dirichlet_alpha_max: float = 0.50,
    ) -> None:
        self.mcts = mcts
        self.temperature = temperature
        self.add_dirichlet_noise = add_dirichlet_noise
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.dirichlet_alpha_min = dirichlet_alpha_min
        self.dirichlet_alpha_max = dirichlet_alpha_max
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

    def get_action(
        self, board: GomokuBoard, return_probs: bool = False, root_noise: bool = False
    ) -> Any:
        """Return a move selected by MCTS.

        If ``return_probs`` is ``True``, also return the visit-count based
        action probabilities produced by the search.
        """
        action_probs = self.mcts.get_action_probs(board, temp=self.temperature)

        if not action_probs:
            # Fallback: pick random move if MCTS failed
            print("[WARNING] MCTS returned no moves. Picking random legal move.")
            legal_moves = board.get_legal_moves()
            selected_action = random.choice(legal_moves)
            if return_probs:
                uniform = 1.0 / len(legal_moves) if legal_moves else 0.0
                probs_dict: Dict[Any, float] = {m: uniform for m in legal_moves}
                return selected_action, probs_dict
            return selected_action

        # === Add Dirichlet noise only on first move if enabled ===
        if self.add_dirichlet_noise and root_noise:
            alpha_eff = self._effective_dirichlet_alpha(board)
            # (Optional) one-time debug per game
            if isinstance(self.dirichlet_alpha, str) and self.dirichlet_alpha == "auto":
                print(
                    f"[Dirichlet] α_eff={alpha_eff:.3f}  ε={self.dirichlet_epsilon:.2f}  "
                    f"|legal|={len(action_probs)}"
                )
            action_probs = self._add_dirichlet_noise(action_probs, alpha=alpha_eff)

        actions, probs = zip(*action_probs.items())

        # Very small negative epsilons can appear from float noise; clamp defensively.
        probs = [max(0.0, float(p)) for p in probs]
        s = sum(probs)
        if s == 0.0:
            # extremely defensive: fall back to uniform over current actions
            probs = [1.0 / len(probs)] * len(probs)
        else:
            # renormalize for numerical safety
            probs = [p / s for p in probs]

        if self.temperature <= 1e-3:
            selected_action = max(action_probs.items(), key=lambda x: x[1])[0]
        else:
            selected_action = random.choices(actions, weights=probs, k=1)[0]

        self.move_number += 1

        if return_probs:
            return selected_action, action_probs
        return selected_action

    def _effective_dirichlet_alpha(self, board: GomokuBoard) -> float:
        # if "auto" or <=0 auto-compute; else user-provided
        if (
            isinstance(self.dirichlet_alpha, str) and self.dirichlet_alpha == "auto"
        ) or (
            isinstance(self.dirichlet_alpha, (int, float)) and self.dirichlet_alpha <= 0
        ):
            n_legal = max(1, len(board.get_legal_moves()))
            alpha = 10.0 / n_legal
            # clip
            alpha = float(
                np.clip(alpha, self.dirichlet_alpha_min, self.dirichlet_alpha_max)
            )
            return alpha
        return float(self.dirichlet_alpha)

    def _add_dirichlet_noise(
        self, action_probs: Dict[Any, float], alpha: float
    ) -> Dict[Any, float]:
        actions = list(action_probs.keys())
        noise = np.random.dirichlet([alpha] * len(actions))
        noisy_probs: Dict[Any, float] = {}
        for a, n in zip(actions, noise):
            noisy_probs[a] = (1 - self.dirichlet_epsilon) * action_probs[
                a
            ] + self.dirichlet_epsilon * n
        return noisy_probs
