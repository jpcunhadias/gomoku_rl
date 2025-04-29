from game.gomoku import GomokuBoard
from game.player import MCTSPlayer, RandomPlayer
from mcts.mcts import MCTS
from model.policy_value_net import PolicyValueNet
from mcts.neural_evaluator import NeuralEvaluator
from cli.play.utils import play_game


def play_mcts_net_vs_random():
    # Create game board
    board = GomokuBoard(board_size=15)

    # Create policy-value network and neural evaluator
    model = PolicyValueNet(board_size=15, num_blocks=5)
    evaluator = NeuralEvaluator(model)
    # Create MCTS using the neural evaluator
    mcts = MCTS(evaluator_fn=evaluator.evaluate, c_puct=1.5, n_simulations=100)

    # Create players
    mcts_player = MCTSPlayer(mcts)
    random_player = RandomPlayer(player_id=2)

    # Play the game
    play_game(board, mcts_player, random_player)

    board.render()
    print("Game Over.")
    winner = board.get_winner()
    print("Winner:", f"Player {winner}" if winner else "Draw")


if __name__ == "__main__":
    play_mcts_net_vs_random()
