from game.gomoku import GomokuBoard
from game.player import MCTSPlayer, RandomPlayer
from mcts.mcts import MCTS
from cli.play.utils import UniformEvaluator, play_game


def play_mcts_pure_vs_random():
    board = GomokuBoard()
    evaluator = UniformEvaluator()
    mcts = MCTS(evaluator_fn=evaluator, c_puct=1.5, n_simulations=100)
    player1 = MCTSPlayer(mcts)
    player2 = RandomPlayer(player_id=2)

    play_game(board, player1, player2)

    print("Game Over.")
    print("Winner:", board.get_winner())


if __name__ == "__main__":
    play_mcts_pure_vs_random()
