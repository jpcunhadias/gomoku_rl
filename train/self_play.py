import os

import numpy as np
import torch
from tqdm import trange

from game import encoder
from game.gomoku import GomokuBoard
from train.augmentation import augment_board_state
from train.replay_buffer import ReplayBuffer


class SelfPlayRunner:
    def __init__(
        self,
        player1,
        player2,
        buffer,
        temperature_schedule=None,
        augment_fn=None,
        verbose=False,
    ):
        self.player1 = player1
        self.player2 = player2
        self.buffer = buffer
        self.temperature_schedule = temperature_schedule or (lambda move: 1.0)
        self.augment_fn = augment_fn
        self.verbose = verbose

    def play_game(self):
        board = GomokuBoard()
        game_data = []
        move_number = 0

        if hasattr(self.player1, "reset"):
            self.player1.reset()
        if hasattr(self.player2, "reset"):
            self.player2.reset()

        while not board.is_terminal():
            current_player = self.player1 if move_number % 2 == 0 else self.player2

            state_tensor = encoder.board_to_tensor(
                board, 1 if move_number % 2 == 0 else -1
            )

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

        augmented_data = []
        for state, policy, value in final_data:
            augmented_state = augment_board_state(state)
            augmented_data.append((augmented_state, policy, value))

        self.buffer.add(augmented_data)

    def _create_pi_from_action(self, board, action):
        """Create a 15x15 policy matrix where the selected move gets probability 1"""
        pi = np.zeros((15, 15), dtype=np.float32)
        if isinstance(action, tuple):
            pi[action[0], action[1]] = 1.0
        else:
            i, j = board.index_to_move(action)
            pi[i, j] = 1.0
        return torch.from_numpy(pi)


def run_selfplay(config, buffer_save_path=None):
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
        augment_fn=augment_board_state,
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
