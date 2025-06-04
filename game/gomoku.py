import numpy as np
from typing import List, Tuple, Optional


class GomokuBoard:
    """Representation of a Gomoku game board."""

    def __init__(self, board_size: int = 15, n_in_row: int = 5) -> None:
        self.board_size = board_size
        self.n_in_row = n_in_row
        self.board = np.zeros(
            (board_size, board_size), dtype=np.int8
        )  # 0=empty, 1=player1, 2=player2
        self.current_player = 1
        self.last_move = None

    def reset(self, start_player: int = 1) -> None:
        """Reset the game board and current player."""
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = start_player
        self.last_move = None

    def get_legal_moves(self) -> List[Tuple[int, int]]:
        """Return coordinates of all empty positions."""
        legal_moves = [
            (r, c)
            for r in range(self.board_size)
            for c in range(self.board_size)
            if self.board[r, c] == 0
        ]
        return legal_moves

    def is_legal_move(self, row: int, col: int) -> bool:
        """Return ``True`` if ``(row, col)`` is empty and on the board."""
        if 0 <= row < self.board_size and 0 <= col < self.board_size:
            return self.board[row, col] == 0
        return False

    def apply_move(self, row: int, col: int) -> None:
        """Apply ``(row, col)`` for the current player."""
        if not self.is_legal_move(row, col):
            raise ValueError(f"Illegal move at ({row}, {col})")

        self.board[row, col] = self.current_player
        self.last_move = (row, col)
        self.current_player = 2 if self.current_player == 1 else 1

    def check_win(self) -> Tuple[bool, int]:
        """Return whether the last move ended the game and the winning player."""

        if self.last_move is None:
            return False, -1

        directions = [
            (1, 0),  # Down
            (0, 1),  # Right
            (1, 1),  # Diagonal down-right
            (1, -1),  # Diagonal down-left
        ]

        last_row, last_col = self.last_move
        player = self.board[last_row, last_col]

        for dr, dc in directions:
            count = 1

            # Check in the positive direction
            r, c = last_row + dr, last_col + dc
            while (
                0 <= r < self.board_size
                and 0 <= c < self.board_size
                and self.board[r, c] == player
            ):
                count += 1
                r += dr
                c += dc

            # Check in the negative direction
            r, c = last_row - dr, last_col - dc
            while (
                0 <= r < self.board_size
                and 0 <= c < self.board_size
                and self.board[r, c] == player
            ):
                count += 1
                r -= dr
                c -= dc

            if count >= self.n_in_row:
                return True, player

        return False, -1

    def check_draw(self) -> bool:
        """Return ``True`` if the board is full and no player has won."""
        if not np.any(self.board == 0):  # no empty cells
            win, _ = self.check_win()
            return not win  # it's a draw if nobody won
        return False

    def get_legal_move_indices(self) -> List[int]:
        """Return legal moves encoded as single indices."""
        return [self.move_to_index(r, c) for (r, c) in self.get_legal_moves()]

    def get_winner(self) -> Optional[int]:
        """Return the winning player or ``None`` if there is no winner."""
        win, winner = self.check_win()
        return winner if win else None

    def get_current_state(self) -> np.ndarray:
        """Return a 4-channel tensor describing the board state."""
        state = np.zeros((4, self.board_size, self.board_size), dtype=np.float32)

        # Player channels
        state[0] = (self.board == self.current_player).astype(np.float32)
        state[1] = (self.board == (2 if self.current_player == 1 else 1)).astype(
            np.float32
        )

        # Last move channel
        if self.last_move:
            r, c = self.last_move
            state[2, r, c] = 1.0

        # Whose turn channel
        state[3, :, :] = 1.0 if self.current_player == 1 else 0.0

        return state

    def render(self) -> None:
        """Print a human readable representation of the board."""
        # Header row with column numbers
        header = "   " + " ".join(f"{c:2d}" for c in range(self.board_size))
        print(header)

        for r in range(self.board_size):
            row_str = f"{r:2d} "  # Row number
            for c in range(self.board_size):
                piece = self.board[r, c]
                if piece == 1:
                    row_str += " X"
                elif piece == 2:
                    row_str += " O"
                else:
                    row_str += " ."
            print(row_str)

        print(f"\nCurrent player: {'X' if self.current_player == 1 else 'O'}")

    def move_to_index(self, row: int, col: int) -> int:
        """Convert ``(row, col)`` into a flat index."""
        return row * self.board_size + col

    def index_to_move(self, index: int) -> Tuple[int, int]:
        """Inverse of :meth:`move_to_index`.``"""
        return divmod(index, self.board_size)

    def clone(self) -> "GomokuBoard":
        """Return a deep copy of the board."""
        new_board = GomokuBoard(board_size=self.board_size, n_in_row=self.n_in_row)
        new_board.board = self.board.copy()  # deep copy of NumPy array
        new_board.current_player = self.current_player
        new_board.last_move = self.last_move
        return new_board

    def is_terminal(self) -> bool:
        """Return ``True`` if the game has ended."""
        win, _ = self.check_win()
        return win or self.check_draw()

    def evaluate_terminal(self) -> float:
        """Evaluate a terminal board from the current player's perspective."""
        win, winner = self.check_win()
        if win:
            return 1.0 if winner == self.current_player else -1.0
        elif self.check_draw():
            return 0.0
        raise RuntimeError("evaluate_terminal() called on non-terminal board.")


class GomokuGameManager:
    """Convenience wrapper that manages a game between two players."""

    def __init__(self, board: Optional[GomokuBoard] = None) -> None:
        self.board = board if board else GomokuBoard()
        self.winner: Optional[int] = None
        self.finished = False

    def reset(self, start_player: int = 1) -> None:
        """Reset the managed game."""
        self.board.reset(start_player=start_player)
        self.winner = None
        self.finished = False

    def play_move(self, row: int, col: int) -> None:
        """Apply a move and update the game state."""
        if self.finished:
            raise RuntimeError("Game is already over.")

        if not self.board.is_legal_move(row, col):
            raise ValueError(f"Illegal move at ({row}, {col})")

        self.board.apply_move(row, col)

        # Check for end state
        win, winner = self.board.check_win()
        if win:
            self.finished = True
            self.winner = winner
        elif self.board.check_draw():
            self.finished = True
            self.winner = None  # Draw

    def play_move_by_index(self, index: int) -> None:
        """Apply a move specified by a flat index."""
        row, col = self.board.index_to_move(index)
        self.play_move(row, col)

    def is_over(self) -> bool:
        """Return ``True`` if the game has finished."""
        return self.finished

    def get_winner(self) -> Optional[int]:
        """Return the winner, or ``None`` for a draw or unfinished game."""
        return self.winner

    def get_current_player(self) -> int:
        """Return the player ID whose turn it is."""
        return self.board.current_player

    def get_legal_moves(self) -> List[Tuple[int, int]]:
        """Return legal moves for the current board."""
        return self.board.get_legal_moves()

    def get_legal_move_indices(self) -> List[int]:
        """Return legal moves encoded as indices."""
        return [self.board.move_to_index(r, c) for (r, c) in self.get_legal_moves()]

    def render(self) -> None:
        """Delegate to :meth:`GomokuBoard.render`."""
        self.board.render()


if __name__ == "__main__":
    pass
