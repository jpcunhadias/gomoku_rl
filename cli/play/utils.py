import logging

from game.gomoku import GomokuBoard
from game.player import MCTSPlayer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _ensure_tuple(board: GomokuBoard, action):
    """Convert index -> (row, col) if needed."""
    if isinstance(action, tuple):
        return action
    return board.index_to_move(action)


def play_game(
    board: GomokuBoard,
    player1,
    player2,
    verbose: bool = False,
    *,
    reset_players: bool = True,
    deterministic_eval: bool = False,  # τ=0, no Dirichlet at root
    force_tuple_action: bool = True,
) -> int:
    """
    Play a game between two players.

    Returns:
        int: Winner (1, 2) or 0 for draw.
    """
    players = [player1, player2]

    # One-time resets for a clean game
    if reset_players:
        for p in players:
            if hasattr(p, "reset"):
                p.reset()
            if isinstance(p, MCTSPlayer) and hasattr(p.mcts, "reset_root"):
                p.mcts.reset_root()

    # For arena: force deterministic (τ=0, no Dirichlet)
    if deterministic_eval:
        for p in players:
            if isinstance(p, MCTSPlayer):
                if hasattr(p, "set_temperature"):
                    p.set_temperature(0.0)
                if hasattr(p, "add_dirichlet_noise"):
                    p.add_dirichlet_noise = False

    current_player = 0
    while not board.is_terminal():
        if verbose:
            board.render()
            current_color = "X" if board.current_player == 1 else "O"
            logging.info(
                f"Current player: {current_color} ({getattr(players[current_player], 'name', 'Player')})"
            )

        player = players[current_player]
        action = player.get_action(board)

        if force_tuple_action:
            action = _ensure_tuple(board, action)

        board.apply_move(*action)

        # Keep both trees in sync
        for p in players:
            if isinstance(p, MCTSPlayer):
                p.mcts.update_with_move(action)

        current_player = 1 - current_player

    if verbose:
        board.render()
        logging.info("Game Over.")

    winner_message = (
        f"Winner: {'Player ' + str(board.get_winner()) if board.get_winner() else 'Draw'}"
    )
    logging.info(winner_message)

    winner = board.get_winner()
    return winner if winner is not None else 0
