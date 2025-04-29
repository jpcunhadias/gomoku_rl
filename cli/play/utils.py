from game.gomoku import GomokuBoard
from game.player import MCTSPlayer


class UniformEvaluator:
    def __call__(self, board):
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            return [], 0.0  # No moves left, return empty list and neutral value
        prior = [(move, 1 / len(legal_moves)) for move in legal_moves]
        value = 0.0  # Assume neutral board
        return prior, value


def play_game(board: GomokuBoard, player1, player2, verbose: bool = False) -> int:
    """
    Play a game between two players.

    Args:
        board (GomokuBoard): The game board instance.
        player1: First player (must have get_action(board)).
        player2: Second player (must have get_action(board)).
        verbose (bool): If True, render board and print moves.

    Returns:
        int: Winner (1, 2, or 0 for draw)
    """
    players = [player1, player2]
    current_player = 0

    while not board.is_terminal():
        if verbose:
            board.render()

        player = players[current_player]
        action = player.get_action(board)

        if verbose:
            print(f"\n{player} plays {action}")

        board.apply_move(*action)

        for p in players:
            if isinstance(p, MCTSPlayer):
                p.mcts.update_with_move(action)

        current_player = 1 - current_player

    if verbose:
        board.render()
        print("Game Over.")
        print(
            "Winner:", f"Player {board.get_winner()}" if board.get_winner() else "Draw"
        )

    return board.get_winner()
