import numpy as np
import torch

from game.gomoku import GomokuBoard


def board_to_tensor(board: GomokuBoard, current_player: int) -> torch.Tensor:
    """Convert a :class:`GomokuBoard` to a PyTorch tensor.

    The tensor has three channels: the current player's stones, the opponent's
    stones and an indicator plane whose elements are all ones.  This format is
    commonly used as input to neural networks.

    Args:
        board: Board to encode.
        current_player: ID of the player who is about to move (1 or 2).

    Returns:
        Tensor of shape ``[3, board_size, board_size]`` representing the board.
    """

    raw_board = board.board
    opponent = 2 if current_player == 1 else 1

    player_plane = (raw_board == current_player).astype(np.float32)
    opponent_plane = (raw_board == opponent).astype(np.float32)
    turn_plane = np.ones_like(raw_board, dtype=np.float32)

    planes = np.stack([player_plane, opponent_plane, turn_plane], axis=0)
    return torch.from_numpy(planes)
