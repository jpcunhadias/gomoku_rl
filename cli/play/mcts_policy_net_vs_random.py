from game.gomoku import GomokuBoard
from game.player import MCTSPlayer, RandomPlayer
from mcts.mcts import MCTS
from model.policy_value_net import PolicyValueNet
from mcts.neural_evaluator import NeuralEvaluator
import random


def play_mcts_net_vs_random():
    # Create game board
    board = GomokuBoard(board_size=15)

    # Create policy-value network and neural evaluator
    model = PolicyValueNet(board_size=15, num_blocks=5)
    evaluator = NeuralEvaluator(model)

    # Create MCTS using the neural evaluator
    mcts = MCTS(evaluator_fn=evaluator.evaluate, c_puct=1.5, n_simulations=100)

    # Create players
    player1 = MCTSPlayer(mcts)
    player2 = RandomPlayer(player_id=2)

    players = [player1, player2]
    current_player = 0

    while not board.is_terminal():
        board.render()

        if isinstance(players[current_player], MCTSPlayer):
            # Special handling for MCTSPlayer
            action_probs = players[current_player].mcts.get_action_probs(board,
                                                                         temp=players[current_player].temperature)

            if not action_probs:
                print("[WARNING] No moves returned by MCTS. Picking random legal move.")
                legal_moves = board.get_legal_moves()
                action = random.choice(legal_moves)
            else:
                action = max(action_probs.items(), key=lambda x: x[1])[0]
        else:
            # For random or human players
            action = players[current_player].get_action(board)

        if isinstance(action, int):
            action = board.index_to_move(action)

        print(f"\n{players[current_player]} plays {action}")
        board.apply_move(*action)
        current_player = 1 - current_player

    board.render()
    print("Game Over.")
    winner = board.get_winner()
    print("Winner:", f"Player {winner}" if winner else "Draw")


if __name__ == "__main__":
    play_mcts_net_vs_random()
