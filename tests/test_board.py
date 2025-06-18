import pytest
import numpy as np
from game.gomoku import GomokuBoard


def test_board_initial_state():
    board = GomokuBoard()
    assert board.board.shape == (8, 8)
    assert np.all(board.board == 0)
    assert board.current_player == 1
    assert board.last_move is None


def test_apply_and_toggle():
    board = GomokuBoard()
    board.apply_move(7, 7)
    assert board.board[7, 7] == 1
    assert board.current_player == 2
    assert board.last_move == (7, 7)

    board.apply_move(7, 8)
    assert board.board[7, 8] == 2
    assert board.current_player == 1


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


def test_current_state_shape():
    board = GomokuBoard()
    board.apply_move(7, 7)
    state = board.get_current_state()
    assert state.shape == (4, 8, 8)
    assert state[0][7][7] == 0.0  # current player is player 2 now
    assert state[1][7][7] == 1.0  # opponent
    assert state[2][7][7] == 1.0  # last move marker
