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
