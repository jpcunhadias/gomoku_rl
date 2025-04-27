from game.player import MCTSPlayer, RandomPlayer
from game.gomoku import GomokuBoard

def dummy_evaluator(board):
    legal_moves = board.get_legal_moves()
    priors = [(move, 1 / len(legal_moves)) for move in legal_moves]
    value = 0  # Stub value
    return priors, value

def play_mcts_vs_random(board, mcts_player, random_player):
    """
    Generic function to play a game between an MCTSPlayer and a RandomPlayer.

    Args:
        board (GomokuBoard): The game board instance.
        mcts_player (MCTSPlayer): Player using MCTS.
        random_player (RandomPlayer): Player playing random moves.
    """
    players = [mcts_player, random_player]
    current_player = 0

    while not board.is_terminal():
        board.render()

        player = players[current_player]
        action = player.get_action(board)
        print(f"\n{player} plays {action}")

        board.apply_move(*action)

        if isinstance(player, MCTSPlayer):
            player.mcts.update_with_move(action)

        current_player = 1 - current_player

    board.render()
    print("Game Over.")
    winner = board.get_winner()
    print("Winner:", f"Player {winner}" if winner else "Draw")
