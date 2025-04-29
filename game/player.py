import random
from typing import Tuple, Protocol, Any
import numpy as np
from game.gomoku import GomokuBoard
from mcts.mcts import MCTS


class Player(Protocol):
    def get_action(self, board: GomokuBoard) -> Tuple[int, int]: ...


class RandomPlayer:
    def __init__(self, player_id: int):
        self.player_id = player_id

    def __repr__(self):
        return f"RandomPlayer({self.player_id})"

    def get_action(self, board: GomokuBoard) -> Tuple[int, int]:
        legal_moves = board.get_legal_moves()
        return random.choice(legal_moves)


class HumanPlayer:
    def __init__(self, player_id: int):
        self.player_id = player_id

    def get_action(self, board: GomokuBoard) -> Tuple[int, int]:
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
    def __init__(self, mcts: MCTS, temperature: float = 1e-3, add_dirichlet_noise=False, dirichlet_alpha=0.3, dirichlet_epsilon=0.25):
        self.mcts = mcts
        self.temperature = temperature
        self.add_dirichlet_noise = add_dirichlet_noise
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.move_number = 0  # Track move number internally

    def __repr__(self):
        return f"MCTSPlayer(temperature={self.temperature})"

    def set_temperature(self, temp: float):
        self.temperature = temp

    def reset(self):
        """Call this before each new game to reset move number."""
        self.move_number = 0

    def get_action(self, board: GomokuBoard) -> Any:
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
            selected_action = actions[0]  # deterministic best move
        else:
            selected_action = random.choices(actions, weights=probs, k=1)[0]

        self.move_number += 1
        return selected_action

    def _add_dirichlet_noise(self, action_probs):
        """Inject Dirichlet noise into action probabilities."""
        actions = list(action_probs.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
        noisy_probs = {}
        for a, n in zip(actions, noise):
            noisy_probs[a] = (1 - self.dirichlet_epsilon) * action_probs[a] + self.dirichlet_epsilon * n
        return noisy_probs

