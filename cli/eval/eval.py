from types import SimpleNamespace
from typing import Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from cli.play.utils import play_game
from game.gomoku import GomokuBoard
from game.player import MCTSPlayer
from mcts.evaluators import NeuralEvaluator, ThreatRolloutEvaluator
from mcts.mcts import MCTS
from model.policy_value_net import PolicyValueNet


def evaluate_model_vs_pure_mcts(
    model: PolicyValueNet,
    device: str,
    config: SimpleNamespace,
    num_games: int = 20,
    board_size: int = 8,
    writer: Optional[SummaryWriter] = None,
    global_step: Optional[int] = None,
) -> float:
    print("[Eval] Starting evaluation against Pure MCTS...")
    model.eval()
    neural_evaluator = NeuralEvaluator(model, device)

    model_wins = 0
    pure_mcts_wins = 0
    draws = 0

    for game_idx in range(num_games):
        board = GomokuBoard(board_size)

        if torch.rand(1).item() < 0.5:
            player1 = MCTSPlayer(
                MCTS(neural_evaluator, config.c_puct, config.eval_num_simulations),
                temperature=config.temperature,
                name="Model",
            )
            player2 = MCTSPlayer(
                MCTS(
                    ThreatRolloutEvaluator(),
                    config.c_puct_pure,
                    config.eval_num_simulations,
                ),
                temperature=1e-3,
                name="MCTS_Pure",
            )
            first_player_is_model = True
        else:
            player1 = MCTSPlayer(
                MCTS(
                    ThreatRolloutEvaluator(),
                    config.c_puct_pure,
                    config.eval_num_simulations,
                ),
                temperature=1e-3,
                name="MCTS_Pure",
            )
            player2 = MCTSPlayer(
                MCTS(neural_evaluator, config.c_puct, config.eval_num_simulations),
                temperature=config.temperature,
                name="Model",
            )
            first_player_is_model = False

        winner = play_game(board, player1, player2, verbose=True)

        if winner == 0:
            draws += 1
        elif (winner == 1 and first_player_is_model) or (
            winner == 2 and not first_player_is_model
        ):
            model_wins += 1
        else:
            pure_mcts_wins += 1

        print(
            f"[Eval] Game {game_idx + 1}/{num_games}: Winner = "
            f"{'Model' if winner in [1, 2] and ((winner == 1 and first_player_is_model) or (winner == 2 and not first_player_is_model)) else 'MCTS_Pure' if winner != 0 else 'Draw'}"
        )

    total = model_wins + pure_mcts_wins + draws
    model_win_rate = model_wins / total if total > 0 else 0.0

    print("\n[Eval] Evaluation Summary:")
    print(f"Model Wins: {model_wins}")
    print(f"MCTS_Pure Wins: {pure_mcts_wins}")
    print(f"Draws: {draws}")
    print(f"Model Win Rate: {model_win_rate:.2f}")

    if writer and global_step is not None:
        writer.add_scalar("Eval/Model_WinRate_vs_PureMCTS", model_win_rate, global_step)

    return model_win_rate
