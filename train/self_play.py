import os
from types import SimpleNamespace
from typing import Callable, List, Tuple, Optional, Any

import numpy as np
import torch
from tqdm import trange

from game import encoder
from game.gomoku import GomokuBoard
from game.player import MCTSPlayer
from mcts.mcts import MCTS
from mcts.evaluators import NeuralEvaluator
from model.policy_value_net import PolicyValueNet
from train.augmentation import augment_data
from train.replay_buffer import ReplayBuffer


class SelfPlayRunner:
    """Orchestrates self-play games between two agents."""

    def __init__(
        self,
        player1,
        player2,
        buffer: ReplayBuffer,
        temperature_schedule: Optional[Callable[[int], float]] = None,
        augment_fn: Optional[
            Callable[
                [List[Tuple[torch.Tensor, torch.Tensor, float]]],
                List[Tuple[torch.Tensor, torch.Tensor, float]],
            ]
        ] = None,
        verbose: bool = False,
    ) -> None:
        self.player1 = player1
        self.player2 = player2
        self.buffer = buffer
        self.temperature_schedule = temperature_schedule or (lambda move: 1.0)
        self.augment_fn = augment_fn
        self.verbose = verbose

    def play_game(self) -> None:
        """Play a single self-play game and store the resulting data."""
        board = GomokuBoard()
        game_data = []
        move_number = 0

        if hasattr(self.player1, "reset"):
            self.player1.reset()
        if hasattr(self.player2, "reset"):
            self.player2.reset()

        while not board.is_terminal():
            current_player = self.player1 if move_number % 2 == 0 else self.player2

            # Encode the board from the perspective of the player whose
            # turn it is. ``board.current_player`` is 1 for player one and
            # 2 for player two, which matches the expected input of
            # ``board_to_tensor``.
            state_tensor = encoder.board_to_tensor(board, board.current_player)

            # Set temperature for MCTS if needed
            if isinstance(current_player, type(self.player1)) and hasattr(
                current_player, "set_temperature"
            ):
                temp = self.temperature_schedule(move_number)
                current_player.set_temperature(temp)

            action = current_player.get_action(board)

            pi = self._create_pi_from_action(board, action)

            game_data.append((state_tensor, pi, 1 if move_number % 2 == 0 else -1))

            board.apply_move(*action)

            if self.verbose:
                board.render()

            move_number += 1

        winner = board.get_winner()
        if winner == 2:
            winner = -1
        elif winner is None:
            winner = 0

        # Encode results as integer classes: 0=loss, 1=draw, 2=win
        final_data = []
        for state_tensor, pi_tensor, player in game_data:
            if winner == 0:
                z = 1
            else:
                z = 2 if winner == player else 0
            final_data.append((state_tensor, pi_tensor, z))

        if self.augment_fn:
            final_data = self.augment_fn(final_data)

        self.buffer.add(final_data)

    def _create_pi_from_action(self, board: GomokuBoard, action: Any) -> torch.Tensor:
        """Create a ``board_size × board_size`` policy tensor with ``1`` at the chosen move."""
        board_size = board.board_size
        pi = np.zeros((board_size, board_size), dtype=np.float32)
        if isinstance(action, tuple):
            pi[action[0], action[1]] = 1.0
        else:
            i, j = board.index_to_move(action)
            pi[i, j] = 1.0
        return torch.from_numpy(pi)


def initialize_model(
    device: str, checkpoint_path: Optional[str] = None
) -> PolicyValueNet:
    model = PolicyValueNet(board_size=8).to(device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading model from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print(
            "No checkpoint found or checkpoint loading skipped. Initialized new PolicyValueNet."
        )
    return model


def create_players(
    evaluator: NeuralEvaluator,
    n_simulations: int,
    config: SimpleNamespace,
) -> Tuple[MCTSPlayer, MCTSPlayer]:
    """Create two MCTS players using parameters from config."""
    player_kwargs = {
        "temperature": config.temperature,
        "add_dirichlet_noise": config.add_dirichlet_noise,
    }

    mcts_kwargs = {
        "evaluator_fn": evaluator,
        "c_puct": config.c_puct,
        "n_simulations": n_simulations,
    }

    player1 = MCTSPlayer(MCTS(**mcts_kwargs), **player_kwargs)
    player2 = MCTSPlayer(MCTS(**mcts_kwargs), **player_kwargs)

    return player1, player2


def run_selfplay_pipeline(
    config: Any,
    load_checkpoint: bool = False,
    buffer_save_path: Optional[str] = None,
) -> Tuple[PolicyValueNet, ReplayBuffer]:
    """Run a full self-play pipeline."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    checkpoint_path = (
        "checkpoints/policy_value_net_best.pth" if load_checkpoint else None
    )
    model = initialize_model(device, checkpoint_path)

    evaluator = NeuralEvaluator(model, device)
    player1, player2 = create_players(
        evaluator, n_simulations=config.self_play_num_simulations, config=config
    )

    if buffer_save_path and os.path.exists(buffer_save_path):
        print(f"Loading existing replay buffer from {buffer_save_path}")
        buffer = ReplayBuffer.load(buffer_save_path)
    else:
        print("No existing buffer found. Initializing new ReplayBuffer.")
        buffer = ReplayBuffer(max_size=config.replay_buffer_size)

    runner = SelfPlayRunner(
        player1=player1,
        player2=player2,
        buffer=buffer,
        temperature_schedule=lambda move: 1.0 if move < 10 else 1e-3,
        augment_fn=augment_data,
        verbose=False,
    )

    for i in trange(config.num_self_play_games, desc="Self-play games"):
        runner.play_game()

    print(f"\nBuffer filled with {len(buffer)} samples.")

    if buffer_save_path:
        os.makedirs(os.path.dirname(buffer_save_path), exist_ok=True)
        buffer.save(buffer_save_path)
        print(f"Replay buffer saved to {buffer_save_path}")

    return model, buffer
