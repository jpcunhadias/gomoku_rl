import torch
import random

from game.gomoku import GomokuBoard
from mcts.mcts import MCTS
from mcts.neural_evaluator import NeuralEvaluator
from model.policy_value_net import PolicyValueNet

class UniformEvaluator:
    def __call__(self, board):
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            return [], 0.0  # No moves left, return empty list and neutral value
        prior = [(move, 1 / len(legal_moves)) for move in legal_moves]
        value = 0.0  # Assume neutral board
        return prior, value


def evaluate_mcts_vs_mcts_pure(
    model_checkpoint_path,
    num_games=20,
    board_size=15,
    num_simulations=100,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load trained model
    model = PolicyValueNet(board_size=board_size)
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Create evaluators
    neural_evaluator = NeuralEvaluator(model, device)
    uniform_evaluator = UniformEvaluator()

    # Helper to create new MCTS players
    def create_mcts_player(evaluator_fn):
        return MCTS(evaluator_fn=evaluator_fn, c_puct=1.0, n_simulations=num_simulations)

    # Stats
    model_wins = 0
    pure_mcts_wins = 0
    draws = 0

    for game_idx in range(num_games):
        board = GomokuBoard(board_size)

        # Randomly assign who is model and who is pure
        if random.random() < 0.5:
            players = [create_mcts_player(neural_evaluator), create_mcts_player(uniform_evaluator)]
            first_player_is_model = True
        else:
            players = [create_mcts_player(uniform_evaluator), create_mcts_player(neural_evaluator)]
            first_player_is_model = False

        current_player_idx = 0

        while not board.is_terminal():
            mcts = players[current_player_idx]
            move_probs = mcts.get_action_probs(board, temp=1e-3)

            if not move_probs:
                print("[WARNING] No moves returned by MCTS during evaluation. Picking random legal move.")
                legal_moves = board.get_legal_moves()
                move = random.choice(legal_moves)
            else:
                move = max(move_probs.items(), key=lambda x: x[1])[0]

            if isinstance(move, int):
                move = board.index_to_move(move)

            board.apply_move(*move)

            players[0].update_with_move(move)
            players[1].update_with_move(move)

            current_player_idx = 1 - current_player_idx

        winner = board.get_winner()

        if winner == 0:
            draws += 1
        elif (winner == 1 and first_player_is_model) or (winner == 2 and not first_player_is_model):
            model_wins += 1
        else:
            pure_mcts_wins += 1

        print(
            f"Game {game_idx + 1}/{num_games}: Winner = {'Model' if (winner == 1 and first_player_is_model) or (winner == 2 and not first_player_is_model) else 'MCTS_Pure' if winner != 0 else 'Draw'}"
        )

    print("\nEvaluation Results:")
    print(f"Model Wins: {model_wins}")
    print(f"MCTS_Pure Wins: {pure_mcts_wins}")
    print(f"Draws: {draws}")
    print(f"Model Win Rate: {model_wins / num_games:.2f}")

if __name__ == "__main__":
    evaluate_mcts_vs_mcts_pure(
        model_checkpoint_path="checkpoints/policy_value_net_epochbest.pth",
        num_games=20,
        board_size=15,
        num_simulations=100,
    )
