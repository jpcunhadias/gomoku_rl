import numpy as np
import torch

from game.encoder import board_to_tensor
from game.gomoku import GomokuBoard


def test_board_to_tensor_shape_and_values():
    board = GomokuBoard()
    board.apply_move(0, 0)  # Player 1
    board.apply_move(0, 1)  # Player 2

    tensor = board_to_tensor(board, current_player=1)
    assert tensor.shape == (3, 8, 8)

    assert tensor[0, 0, 0] == 1.0  # player 1
    assert tensor[1, 0, 1] == 1.0  # player 2
    assert torch.all(tensor[2] == 1.0)  # current player's turn


def test_board_to_tensor_player2_view():
    board = GomokuBoard()
    board.apply_move(0, 0)  # Player 1
    board.apply_move(0, 1)  # Player 2

    tensor = board_to_tensor(board, current_player=2)

    assert tensor.shape == (3, 8, 8)
    # From player 2 perspective the second column contains the player's stone
    assert tensor[0, 0, 1] == 1.0  # player 2 plane
    assert tensor[1, 0, 0] == 1.0  # opponent plane
    assert torch.all(tensor[2] == 1.0)


def test_board_to_tensor_different_board_size():
    board = GomokuBoard(board_size=15)
    board.apply_move(0, 0)  # Player 1
    board.apply_move(14, 14)  # Player 2

    tensor = board_to_tensor(board, current_player=1)
    assert tensor.shape == (3, 15, 15)

    assert tensor[0, 0, 0] == 1.0  # player 1
    assert tensor[1, 14, 14] == 1.0  # player 2
    assert torch.all(tensor[2] == 1.0)


def test_board_to_tensor_empty_board():
    board = GomokuBoard(board_size=5)
    tensor = board_to_tensor(board, current_player=1)

    assert tensor.shape == (3, 5, 5)
    assert torch.all(tensor[0] == 0.0)
    assert torch.all(tensor[1] == 0.0)
    assert torch.all(tensor[2] == 1.0)


def test_board_to_tensor_full_board():
    board = GomokuBoard(board_size=3)
    board.board = np.array([[1, 2, 1], [2, 1, 2], [1, 2, 1]])

    tensor = board_to_tensor(board, current_player=1)

    assert tensor.shape == (3, 3, 3)

    expected_player1_plane = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
    expected_player2_plane = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]

    assert torch.all(tensor[0] == torch.tensor(expected_player1_plane, dtype=torch.float32))
    assert torch.all(tensor[1] == torch.tensor(expected_player2_plane, dtype=torch.float32))
    assert torch.all(tensor[2] == 1.0)
