from game.gomoku import GomokuBoard
from game.player import MCTSPlayer, RandomPlayer
from mcts.mcts import MCTS
from cli.play.utils import play_game
from mcts.evaluators import ThreatRolloutEvaluator


def play_mcts_pure_vs_random() -> None:
    """Play a game between a pure MCTS agent and a random agent."""
    board = GomokuBoard()
    mcts = MCTS(evaluator_fn=ThreatRolloutEvaluator(), c_puct=1.5, n_simulations=3200)
    player1 = MCTSPlayer(mcts)
    player2 = RandomPlayer(player_id=2)

    play_game(board, player1, player2, verbose=True)

    print("Game Over.")
    print("Winner:", board.get_winner())


if __name__ == "__main__":
    play_mcts_pure_vs_random()
