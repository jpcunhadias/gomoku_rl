import random
from typing import Optional

from torch import nn
from torch.utils.tensorboard import SummaryWriter

from cli.play.utils import UniformEvaluator, play_game
from game.gomoku import GomokuBoard
from game.player import MCTSPlayer
from mcts.mcts import MCTS
from mcts.neural_evaluator import NeuralEvaluator


def evaluate_model_vs_pure_mcts(
        model: nn.Module,
        device: str,
        num_games: int = 20,
        board_size: int = 15,
        num_simulations: int = 100,
        writer: Optional[SummaryWriter] = None,
        global_step: Optional[int] = None,
) -> float:
    """
    Evaluate the model against a pure MCTS agent.

    Args:
        model (nn.Module): Trained PolicyValueNet.
        device (str): Device ("cuda" or "cpu").
        num_games (int): Number of evaluation games.
        board_size (int): Size of the Gomoku board.
        num_simulations (int): Number of MCTS simulations per move.
        writer (SummaryWriter, optional): TensorBoard writer.
        global_step (int, optional): Current training step for logging.

    Returns:
        float: Model's win rate against pure MCTS.
    """
    print("[Eval] Starting evaluation against Pure MCTS...")

    # Create evaluators
    model.eval()
    neural_evaluator = NeuralEvaluator(model, device)
    uniform_evaluator = UniformEvaluator()

    model_wins = 0
    pure_mcts_wins = 0
    draws = 0

    for game_idx in range(num_games):
        board = GomokuBoard(board_size)

        # Randomly assign sides
        if random.random() < 0.5:
            player1 = MCTSPlayer(
                MCTS(neural_evaluator, n_simulations=num_simulations),
                temperature=1e-3,
            )
            player2 = MCTSPlayer(
                MCTS(uniform_evaluator, n_simulations=num_simulations),
                temperature=1e-3,
            )
            first_player_is_model = True
        else:
            player1 = MCTSPlayer(
                MCTS(uniform_evaluator, n_simulations=num_simulations),
                temperature=1e-3,
            )
            player2 = MCTSPlayer(
                MCTS(neural_evaluator, n_simulations=num_simulations),
                temperature=1e-3,
            )
            first_player_is_model = False

        winner = play_game(board, player1, player2)

        if winner == 0:
            draws += 1
            winner_str = "Draw"
        elif (winner == 1 and first_player_is_model) or (winner == 2 and not first_player_is_model):
            model_wins += 1
            winner_str = "Model"
        else:
            pure_mcts_wins += 1
            winner_str = "MCTS_Pure"

        print(f"[Eval] Game {game_idx + 1}/{num_games}: Winner = {winner_str}")

    total_games = model_wins + pure_mcts_wins + draws
    model_win_rate = model_wins / total_games if total_games > 0 else 0.0

    print("\n[Eval] Evaluation Summary:")
    print(f"Model Wins: {model_wins}")
    print(f"MCTS_Pure Wins: {pure_mcts_wins}")
    print(f"Draws: {draws}")
    print(f"Model Win Rate: {model_win_rate:.2f}")

    # Optional TensorBoard logging
    if writer and global_step is not None:
        writer.add_scalar("Eval/Model_WinRate_vs_PureMCTS", model_win_rate, global_step)

    return model_win_rate