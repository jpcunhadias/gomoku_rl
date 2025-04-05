import random
from typing import Tuple, Protocol

from game.gomoku import GomokuBoard


class Player(Protocol):
    def get_action(self, board: GomokuBoard) -> Tuple[int, int]:
        ...


class RandomPlayer:
    def __init__(self, player_id: int):
        self.player_id = player_id

    def get_action(self, board: GomokuBoard) -> Tuple[int, int]:
        legal_moves = board.get_legal_moves()
        return random.choice(legal_moves)

    def __repr__(self):
        return f"RandomPlayer({self.player_id})"


class HumanPlayer:
    def __init__(self, player_id: int):
        self.player_id = player_id

    def get_action(self, board: GomokuBoard) -> Tuple[int, int]:
        while True:
            try:
                move_str = input(f"Player {self.player_id}, enter your move as 'row,col': ")
                row, col = map(int, move_str.strip().split(","))
                if board.is_legal_move(row, col):
                    return (row, col)
                else:
                    print("Illegal move. Try again.")
            except Exception:
                print("Invalid input. Format must be: row,col (e.g. 7,7)")
