import os
import numpy as np
import torch
from typing import Callable, List, Tuple, Optional, Any
from tqdm import trange

from game import encoder
from game.gomoku import GomokuBoard
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
        augment_fn: Optional[Callable[[List[Tuple[torch.Tensor, torch.Tensor, float]]], List[Tuple[torch.Tensor, torch.Tensor, float]]]] = None,
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

        final_data = []
        for state_tensor, pi_tensor, player in game_data:
            z = 1.0 if winner == player else 0.0
            final_data.append((state_tensor, pi_tensor, z))

        if self.augment_fn:
            final_data = self.augment_fn(final_data)

        self.buffer.add(final_data)

    def _create_pi_from_action(self, board: GomokuBoard, action: Any) -> torch.Tensor:
        """Create a ``15×15`` policy tensor with ``1`` at the chosen move."""
        pi = np.zeros((15, 15), dtype=np.float32)
        if isinstance(action, tuple):
            pi[action[0], action[1]] = 1.0
        else:
            i, j = board.index_to_move(action)
            pi[i, j] = 1.0
        return torch.from_numpy(pi)


def run_selfplay(config: Any, buffer_save_path: Optional[str] = None) -> Tuple[Any, ReplayBuffer]:
    """Run multiple self-play games and optionally save the buffer."""
    from model.policy_value_net import PolicyValueNet
    from mcts.mcts import MCTS
    from mcts.neural_evaluator import NeuralEvaluator
    from game.player import MCTSPlayer

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = PolicyValueNet(board_size=15).to(device)
    evaluator = NeuralEvaluator(model, device)

    # Create players
    player1 = MCTSPlayer(
        MCTS(evaluator_fn=evaluator, c_puct=1.5, n_simulations=config.self_play_num_simulations),
        temperature=1.0, add_dirichlet_noise=True
    )
    player2 = MCTSPlayer(
        MCTS(evaluator_fn=evaluator, c_puct=1.5, n_simulations=config.self_play_num_simulations),
        temperature=1.0, add_dirichlet_noise=True
    )

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
