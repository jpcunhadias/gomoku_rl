from __future__ import annotations

import logging
import random
from enum import Enum, auto
from typing import Any, Dict, List, Protocol, Tuple

import numpy as np

from game.gomoku import GomokuBoard
from mcts.mcts import MCTS

# Configure logging for the module
logger = logging.getLogger(__name__)


class DirichletAlphaMode(Enum):
    AUTO = auto()
    FIXED = auto()


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
        dirichlet_alpha_mode: DirichletAlphaMode = DirichletAlphaMode.AUTO,
        dirichlet_alpha_fixed: float = 0.15,  # used if mode == FIXED
        dirichlet_epsilon: float = 0.25,
        name: str = "MCTSPlayer",
        dirichlet_alpha_min: float = 0.02,
        dirichlet_alpha_max: float = 0.50,
    ) -> None:
        self.mcts = mcts
        self.temperature = temperature
        self.add_dirichlet_noise = add_dirichlet_noise

        self.dirichlet_alpha_mode = dirichlet_alpha_mode
        self.dirichlet_alpha_fixed = float(dirichlet_alpha_fixed)
        self.dirichlet_epsilon = float(dirichlet_epsilon)
        self.dirichlet_alpha_min = float(dirichlet_alpha_min)
        self.dirichlet_alpha_max = float(dirichlet_alpha_max)

        self.move_number = 0
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
            logger.warning("MCTS returned no moves. Picking random legal move.")
            legal_moves = board.get_legal_moves()
            selected_action = random.choice(legal_moves)
            if return_probs:
                uniform = 1.0 / len(legal_moves) if legal_moves else 0.0
                probs_dict: Dict[Any, float] = {m: uniform for m in legal_moves}
                return selected_action, probs_dict
            return selected_action

        # Root Dirichlet noise (one-time, at root only)
        if self.add_dirichlet_noise and root_noise:
            alpha_eff = self.get_dirichlet_alpha(board)
            logger.debug(
                "[Dirichlet] alpha_eff=%.3f  eps=%.2f  |legal|=%d",
                alpha_eff,
                self.dirichlet_epsilon,
                len(action_probs),
            )
            action_probs = self._add_dirichlet_noise(action_probs, alpha=alpha_eff)

        actions, probs = zip(*action_probs.items())
        probs = self._normalize_probabilities(list(probs))

        if self.temperature <= 1e-3:
            selected_action = max(action_probs.items(), key=lambda x: x[1])[0]
        else:
            selected_action = random.choices(actions, weights=probs, k=1)[0]

        self.move_number += 1
        return (selected_action, action_probs) if return_probs else selected_action

    @staticmethod
    def _normalize_probabilities(probs: List[float]) -> List[float]:
        """Clamp negatives, renormalize, uniform fallback if degenerate."""
        probs = [max(0.0, float(p)) for p in probs]
        s = sum(probs)
        if s <= 1e-12:
            return [1.0 / len(probs)] * len(probs) if probs else []
        return [p / s for p in probs]

    def get_dirichlet_alpha(self, board: GomokuBoard) -> float:
        if self.dirichlet_alpha_mode is DirichletAlphaMode.FIXED:
            return float(self.dirichlet_alpha_fixed)
        # AUTO: alpha ≈ 10 / n_legal, clipped
        n_legal = max(1, len(board.get_legal_moves()))
        alpha = 10.0 / n_legal
        return float(np.clip(alpha, self.dirichlet_alpha_min, self.dirichlet_alpha_max))

    def _add_dirichlet_noise(
        self, action_probs: Dict[Any, float], alpha: float
    ) -> Dict[Any, float]:
        actions = list(action_probs.keys())
        noise = np.random.dirichlet([alpha] * len(actions))
        # convex mix keeps sum==1 if inputs sum==1
        return {
            a: (1 - self.dirichlet_epsilon) * action_probs[a]
            + self.dirichlet_epsilon * n
            for a, n in zip(actions, noise)
        }
