import torch

from game.encoder import board_to_tensor
from game.gomoku import GomokuBoard


def test_board_to_tensor_shape_and_values():
    board = GomokuBoard()
    board.apply_move(0, 0)  # Player 1
    board.apply_move(0, 1)  # Player 2

    tensor = board_to_tensor(board, current_player=1)
    assert tensor.shape == (3, 15, 15)

    assert tensor[0, 0, 0] == 1.0  # player 1
    assert tensor[1, 0, 1] == 1.0  # player 2
    assert torch.all(tensor[2] == 1.0)  # current player's turn


def test_board_to_tensor_player2_view():
    board = GomokuBoard()
    board.apply_move(0, 0)  # Player 1
    board.apply_move(0, 1)  # Player 2

    tensor = board_to_tensor(board, current_player=2)

    assert tensor.shape == (3, 15, 15)
    # From player 2 perspective the second column contains the player's stone
    assert tensor[0, 0, 1] == 1.0  # player 2 plane
    assert tensor[1, 0, 0] == 1.0  # opponent plane
    assert torch.all(tensor[2] == 1.0)
