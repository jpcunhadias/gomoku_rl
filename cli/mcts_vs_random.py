from game.gomoku import GomokuBoard
from game.player import MCTSPlayer, RandomPlayer
from mcts.mcts import MCTS


def dummy_evaluator(board):
    legal_moves = board.get_legal_moves()
    priors = [(move, 1 / len(legal_moves)) for move in legal_moves]
    value = 0  # Stub value
    return priors, value


def play_mcts_vs_random():
    board = GomokuBoard()
    mcts = MCTS(evaluator_fn=dummy_evaluator, c_puct=1.5, n_simulations=100)
    player1 = MCTSPlayer(mcts)
    player2 = RandomPlayer(player_id=2)

    players = [player1, player2]
    current_player = 0

    while not board.is_terminal():
        print(board)
        action = players[current_player].get_action(board)
        print(f"\n{players[current_player]} plays {action}")
        board.apply_move(*action)
        current_player = 1 - current_player

    print(board)
    print("Game Over.")
    print("Winner:", board.get_winner())


if __name__ == "__main__":
    play_mcts_vs_random()
