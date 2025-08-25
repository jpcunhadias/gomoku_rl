import os
from types import SimpleNamespace
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import torch
from tqdm import trange

from game import encoder
from game.gomoku import GomokuBoard
from game.player import MCTSPlayer
from mcts.evaluators import NeuralEvaluator
from mcts.mcts import MCTS
from model.policy_value_net import PolicyValueNet
from train.augmentation import augment_data
from train.distributions import entropy_over_legal, kl_over_legal
from train.replay_buffer import ReplayBuffer
from train.sample_logger import SampleLogger
from train.schema import SampleV2


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
        self.logger = SampleLogger("checkpoints/selfplay_v2.jsonl")

    def play_game(self) -> None:
        """Play a single self-play game and store the resulting data."""
        board = GomokuBoard()
        game_data = []
        move_number = 0
        recs_this_game = []

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

            action, visit_probs = current_player.get_action(board, return_probs=True)

            board_size = board.board_size
            pi_arr = np.zeros((board_size, board_size), dtype=np.float32)
            for move, prob in visit_probs.items():
                if isinstance(move, tuple):
                    r, c = move
                else:
                    r, c = board.index_to_move(move)
                pi_arr[r, c] = prob
            pi = torch.from_numpy(pi_arr)

            legal_mask = torch.zeros(board_size, board_size, dtype=torch.bool)
            for r, c in board.get_legal_moves():
                legal_mask[r, c] = True

            # Optional diagnostics: run net once on this state to get net policy
            with torch.no_grad():
                logits, _ = current_player.mcts.evaluator_fn.model(
                    state_tensor.unsqueeze(0).to(
                        current_player.mcts.evaluator_fn.device
                    )
                )
                net_pi = (
                    torch.softmax(logits.view(-1), dim=0)
                    .view(board_size, board_size)
                    .cpu()
                )

            h_mcts = entropy_over_legal(pi, legal_mask)
            h_net = entropy_over_legal(net_pi, legal_mask)
            kl_nm = kl_over_legal(net_pi, pi, legal_mask)

            rec = SampleV2(
                state=state_tensor,
                pi_mcts=pi,
                v_scalar=0.0,  # Fill v at game end if you log per-move; or log at end
                legal_mask=legal_mask,
                move_number=move_number,
                sims=self.player1.mcts.n_simulations
                if hasattr(self.player1.mcts, "n_simulations")
                else -1,
                tau=getattr(current_player, "temperature", 1.0),
                c_puct=getattr(self.player1.mcts, "c_puct", 1.5),
                dirichlet_alpha=getattr(current_player, "dirichlet_alpha", 0.3),
                dirichlet_eps=getattr(current_player, "dirichlet_epsilon", 0.25),
                entropy_pi_mcts=h_mcts,
                entropy_pi_net=h_net,
                kl_net_mcts=kl_nm,
                symmetry_id=0,  # Fill properly if you record which transform you applied later
                canon_hash=None,
            )

            game_data.append((state_tensor, pi, 1 if move_number % 2 == 0 else -1))

            # Apply the selected move and advance
            if not isinstance(action, tuple):
                action = board.index_to_move(action)
            board.apply_move(*action)

            if self.verbose:
                board.render()

            move_number += 1

            # Write as a dict (dataclasses.asdict is fine too)
            self.logger.write(rec.__dict__)

        winner = board.get_winner()
        if winner == 2:
            winner = -1
        elif winner is None:
            winner = 0

        # Encode results as scalar: -1=loss, 0=draw, 1=win
        final_data = []
        for (state_tensor, pi_tensor, player_sign), rec in zip(
            game_data, recs_this_game
        ):
            z = 0.0 if winner == 0 else (1.0 if winner == player_sign else -1.0)
            final_data.append((state_tensor, pi_tensor, z))
            rec.v_scalar = z
            self.logger.write(rec.__dict__)  # write once per move now that z is known

        if self.augment_fn:
            final_data = self.augment_fn(final_data)

        self.buffer.add(final_data)


def initialize_model(
    device: str, checkpoint_path: Optional[str] = None
) -> PolicyValueNet:
    model = PolicyValueNet(board_size=8).to(device)
    model._init_weights()
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
        "use_rave": config.use_rave,
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
