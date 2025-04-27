from game.gomoku import GomokuBoard
from game.player import MCTSPlayer, RandomPlayer
from mcts.mcts import MCTS
from cli.play.utils import dummy_evaluator, play_mcts_vs_random




def play_mcts_pure_vs_random():
    board = GomokuBoard()
    mcts = MCTS(evaluator_fn=dummy_evaluator, c_puct=1.5, n_simulations=100)
    player1 = MCTSPlayer(mcts)
    player2 = RandomPlayer(player_id=2)

    play_mcts_vs_random(board, player1, player2)

    print("Game Over.")
    print("Winner:", board.get_winner())


if __name__ == "__main__":
    play_mcts_pure_vs_random()
