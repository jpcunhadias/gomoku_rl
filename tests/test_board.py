import numpy as np
import pytest

from game.gomoku import GomokuBoard


def test_board_initial_state():
    board = GomokuBoard()
    assert board.board.shape == (8, 8)
    assert np.all(board.board == 0)
    assert board.current_player == 1
    assert board.last_move is None


def test_apply_and_toggle():
    board = GomokuBoard()  # Default board_size=8
    board.apply_move(7, 6)
    assert board.board[7, 6] == 1
    assert board.current_player == 2
    assert board.last_move == (7, 6)

    board.apply_move(7, 7)
    assert board.board[7, 7] == 2
    assert board.current_player == 1
    assert board.last_move == (7, 7)


def test_illegal_move():
    board = GomokuBoard()
    board.apply_move(7, 7)
    with pytest.raises(ValueError):
        board.apply_move(7, 7)  # already occupied

    with pytest.raises(ValueError):
        board.apply_move(8, 8)  # out of bounds


def test_win_horizontal():
    board = GomokuBoard()
    for c in range(5):
        board.board[7, c] = 1
    board.last_move = (7, 4)  # assume last move was placed
    board.current_player = 2  # simulate turn switch

    win, winner = board.check_win()
    assert win is True
    assert winner == 1


def test_win_vertical():
    board = GomokuBoard()
    for r in range(5):
        board.board[r, 0] = 1
    board.last_move = (4, 0)
    board.current_player = 2

    win, winner = board.check_win()
    assert win is True
    assert winner == 1


def test_win_diag1():
    board = GomokuBoard()
    for i in range(5):
        board.board[i, i] = 1
    board.last_move = (4, 4)
    board.current_player = 2

    win, winner = board.check_win()
    assert win is True
    assert winner == 1


def test_win_diag2():
    board = GomokuBoard()
    for i in range(5):
        board.board[i, 4 - i] = 1
    board.last_move = (4, 0)
    board.current_player = 2

    win, winner = board.check_win()
    assert win is True
    assert winner == 1


def test_current_state_shape():
    board = GomokuBoard()
    board.apply_move(7, 7)
    state = board.get_current_state()
    assert state.shape == (4, 8, 8)
    assert state[0][7][7] == 0.0  # current player is player 2 now
    assert state[1][7][7] == 1.0  # opponent
    assert state[2][7][7] == 1.0  # last move marker


def test_win_player2():
    board = GomokuBoard()
    for c in range(5):
        board.board[0, c] = 2
    board.last_move = (0, 4)
    board.current_player = 1

    win, winner = board.check_win()
    assert win is True
    assert winner == 2


def test_draw():
    board = GomokuBoard(board_size=3)
    board.board = np.array([[1, 2, 1], [1, 2, 2], [2, 1, 1]])
    board.last_move = (0, 2)
    board.current_player = 2

    win, winner = board.check_win()
    assert win is False

    # Check if the board is full
    assert np.all(board.board != 0)

    # In a draw situation, is_terminal() should be true, and get_winner() should be None
    assert board.is_terminal() is True
    assert board.get_winner() is None


def test_get_legal_moves():
    board = GomokuBoard(board_size=3)
    board.apply_move(0, 0)
    board.apply_move(1, 1)
    board.apply_move(2, 2)

    legal_moves = board.get_legal_moves()
    expected_moves = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]

    assert len(legal_moves) == len(expected_moves)
    assert set(legal_moves) == set(expected_moves)
