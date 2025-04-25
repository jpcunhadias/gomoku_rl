import os
import random

import numpy as np
import torch
from tqdm import trange

from game import encoder
from game.gomoku import GomokuBoard
from mcts.mcts import MCTS
from mcts.neural_evaluator import NeuralEvaluator
from mcts.tree_node import TreeNode
from model.policy_value_net import PolicyValueNet
from train.replay_buffer import ReplayBuffer
from train.augmentation import augment_board_state


class SelfPlayRunner:
    def __init__(
            self,
            game_cls,
            mcts_cls,
            evaluator,
            buffer,
            num_simulations=800,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
            temperature_schedule=None,
            augment_fn=None,
            verbose=False,
    ):
        self.game_cls = game_cls
        self.mcts_cls = mcts_cls
        self.evaluator = evaluator
        self.buffer = buffer
        self.num_simulations = num_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.temperature_schedule = temperature_schedule or (lambda move: 1.0)
        self.augment_fn = augment_fn
        self.verbose = verbose

    def play_game(self):
        board = self.game_cls()
        mcts = self.mcts_cls(
            evaluator_fn=self.evaluator, n_simulations=self.num_simulations
        )
        mcts.root = TreeNode()

        game_data = []
        move_number = 0
        current_player = 1
        first_move = True

        while not board.is_terminal():
            state_tensor = encoder.board_to_tensor(board, current_player)
            temperature = self.temperature_schedule(move_number)

            action_probs = mcts.get_action_probs(board, temp=temperature)

            if first_move:
                action_probs = self._add_dirichlet_noise(action_probs)
                first_move = False

            pi = self._dict_to_policy_vector(action_probs, board.get_legal_moves())
            game_data.append((state_tensor, pi, current_player))

            actions, probs = zip(*action_probs.items())
            move = random.choices(actions, weights=probs, k=1)[0]

            board.apply_move(*move)
            mcts.update_with_move(move)

            if self.verbose:
                board.render()

            move_number += 1
            current_player *= -1

        winner = board.get_winner()
        if winner == 2:
            winner = -1

        final_data = []

        for state_tensor, pi, player in game_data:
            z = 1.0 if winner == player else 0.0
            pi_tensor = torch.from_numpy(pi)
            final_data.append((state_tensor, pi_tensor, z))

        if self.augment_fn:
            final_data = self.augment_fn(final_data)

        augmented_data = []
        for i, (state, policy, value) in enumerate(final_data):
            augmented_state = augment_board_state(state)
            augmented_data.append((augmented_state, policy, value))

            # DEBUGGING: Uncomment to visualize augmented states
            # if i < 3:
            #     print(f"[Dry-Run] Sample {i}: Augmented State Shape: {augmented_state.shape}")
            #     print(augmented_state[0])  # Print first channel for quick visual check

        self.buffer.add(augmented_data)

    def _add_dirichlet_noise(self, action_probs):
        actions = list(action_probs.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
        noisy_probs = {}
        for a, n in zip(actions, noise):
            noisy_probs[a] = (1 - self.dirichlet_epsilon) * action_probs[
                a
            ] + self.dirichlet_epsilon * n
        return noisy_probs

    def _dict_to_policy_vector(self, action_probs, legal_moves):
        """Converts action_probs to a 15x15 π vector (matching network output)"""
        pi = np.zeros((15, 15), dtype=np.float32)
        for (i, j), prob in action_probs.items():
            pi[i, j] = prob
        return pi


def run_selfplay(config, num_games=50, mcts_simulations=800, buffer_save_path=None):
    # Initialize model and evaluator
    model = PolicyValueNet(board_size=15)
    evaluator = NeuralEvaluator(model)

    # Initialize replay buffer
    buffer = ReplayBuffer(max_size=config.replay_buffer_size)

    # Set up runner
    runner = SelfPlayRunner(
        game_cls=GomokuBoard,
        mcts_cls=MCTS,
        evaluator=evaluator,
        buffer=buffer,
        num_simulations=mcts_simulations,
        temperature_schedule=lambda move: 1.0 if move < 10 else 1e-3,
        verbose=False,
    )

    for i in trange(num_games, desc="Self-play games"):
        runner.play_game()

    print(f"\nBuffer filled with {len(buffer)} samples")

    if buffer_save_path:
        os.makedirs(os.path.dirname(buffer_save_path), exist_ok=True)
        buffer.save(buffer_save_path)
        print(f"Replay buffer saved to {buffer_save_path}")

    return model, buffer
