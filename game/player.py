import random
from typing import Tuple, Protocol, Any

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
    def __init__(self, mcts: MCTS, temperature: float = 1e-3):
        self.mcts = mcts
        self.temperature = temperature

    def __repr__(self):
        return f"MCTSPlayer(temperature={self.temperature})"

    def get_action(self, board: GomokuBoard) -> Any:
        action_probs = self.mcts.get_action_probs(board, temp=self.temperature)

        if not action_probs:
            # Fallback: pick random move if MCTS failed
            print("[WARNING] MCTS returned no moves. Picking random legal move.")
            legal_moves = board.get_legal_moves()
            return random.choice(legal_moves)

        actions, probs = zip(*action_probs.items())

        if self.temperature <= 1e-3:
            return actions[0]  # deterministic best move
        else:
            return random.choices(actions, weights=probs, k=1)[0]
