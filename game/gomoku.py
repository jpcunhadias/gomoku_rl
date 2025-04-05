import numpy as np


class GomokuBoard:
    def __init__(self, board_size=15, n_in_row=5):
        self.board_size = board_size
        self.n_in_row = n_in_row
        self.board = np.zeros((board_size, board_size), dtype=np.int8)  # 0=empty, 1=player1, 2=player2
        self.current_player = 1
        self.last_move = None

    def reset(self, start_player=1):
        """Reset the game board and player to the starting state."""
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = start_player
        self.last_move = None

    def get_legal_moves(self):
        """Return a list of (row, col) tuples where the board is empty."""
        legal_moves = [
            (r, c)
            for r in range(self.board_size)
            for c in range(self.board_size)
            if self.board[r, c] == 0
        ]
        return legal_moves

    def is_legal_move(self, row, col):
        """Return True if the position is on the board and unoccupied."""
        if 0 <= row < self.board_size and 0 <= col < self.board_size:
            return self.board[row, col] == 0
        return False

    def apply_move(self, row, col):
        """Apply a move for the current player at the given position."""
        if not self.is_legal_move(row, col):
            raise ValueError(f"Illegal move at ({row}, {col})")

        self.board[row, col] = self.current_player
        self.last_move = (row, col)
        self.current_player = 2 if self.current_player == 1 else 1

    def check_win(self):
        if self.last_move is None:
            return False, -1

        directions = [
            (1, 0),  # Down
            (0, 1),  # Right
            (1, 1),  # Diagonal down-right
            (1, -1)  # Diagonal down-left
        ]

        last_row, last_col = self.last_move
        player = self.board[last_row, last_col]

        for dr, dc in directions:
            count = 1

            # Check in the positive direction
            r, c = last_row + dr, last_col + dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                count += 1
                r += dr
                c += dc

            # Check in the negative direction
            r, c = last_row - dr, last_col - dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                count += 1
                r -= dr
                c -= dc

            if count >= self.n_in_row:
                return True, player

        return False, -1

    def check_draw(self):
        """Return True if the game is a draw (board full and no winner)."""
        if not np.any(self.board == 0):  # no empty cells
            win, _ = self.check_win()
            return not win  # it's a draw if nobody won
        return False

    def get_winner(self):
        win, winner = self.check_win()
        return winner if win else None

    def get_current_state(self):
        """Return a 4-channel tensor of the board state from current player's perspective."""
        state = np.zeros((4, self.board_size, self.board_size), dtype=np.float32)

        # Player channels
        state[0] = (self.board == self.current_player).astype(np.float32)
        state[1] = (self.board == (2 if self.current_player == 1 else 1)).astype(np.float32)

        # Last move channel
        if self.last_move:
            r, c = self.last_move
            state[2, r, c] = 1.0

        # Whose turn channel
        state[3, :, :] = 1.0 if self.current_player == 1 else 0.0

        return state

    def render(self):
        """Print a nicely aligned view of the board."""
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

    def move_to_index(self, row, col):
        return row * self.board_size + col

    def index_to_move(self, index):
        return divmod(index, self.board_size)  # returns (row, col)

    def clone(self):
        """Return a deep copy of the current board state."""
        new_board = GomokuBoard(board_size=self.board_size, n_in_row=self.n_in_row)
        new_board.board = self.board.copy()  # deep copy of NumPy array
        new_board.current_player = self.current_player
        new_board.last_move = self.last_move
        return new_board

    def is_terminal(self):
        win, _ = self.check_win()
        return win or self.check_draw()

    def evaluate_terminal(self):
        win, winner = self.check_win()
        if win:
            return 1.0 if winner == self.current_player else -1.0
        elif self.check_draw():
            return 0.0
        raise RuntimeError("evaluate_terminal() called on non-terminal board.")


class GomokuGameManager:
    def __init__(self, board=None):
        self.board = board if board else GomokuBoard()
        self.winner = None
        self.finished = False

    def reset(self, start_player=1):
        self.board.reset(start_player=start_player)
        self.winner = None
        self.finished = False

    def play_move(self, row, col):
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

    def play_move_by_index(self, index):
        row, col = self.board.index_to_move(index)
        self.play_move(row, col)

    def is_over(self):
        return self.finished

    def get_winner(self):
        return self.winner

    def get_current_player(self):
        return self.board.current_player

    def get_legal_moves(self):
        return self.board.get_legal_moves()

    def get_legal_move_indices(self):
        return [self.board.move_to_index(r, c) for (r, c) in self.get_legal_moves()]

    def render(self):
        self.board.render()


if __name__ == "__main__":
    pass
