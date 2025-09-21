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
from train.canonicalize import canonicalize_state, minhash_symmetries
from train.distributions import entropy_over_legal, kl_over_legal
from train.diversity_manager import DiversityManager
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
        augment_fn: Optional[
            Callable[
                [List[Tuple[torch.Tensor, torch.Tensor, float]]],
                List[Tuple[torch.Tensor, torch.Tensor, float]],
            ]
        ] = None,
        verbose: bool = False,
        diversity_manager: Optional[DiversityManager] = None,
        config: Optional[SimpleNamespace] = None,
    ) -> None:
        self.player1 = player1
        self.player2 = player2
        self.buffer = buffer
        self.augment_fn = augment_fn
        self.verbose = verbose
        self.logger = SampleLogger("checkpoints/selfplay_v2.jsonl")
        self.div_manager = diversity_manager
        self.config = config

        if self.config is not None:
            self.tau_cutoff_plies = getattr(config, "tau_cutoff_plies", 12)
            self.phase_cutoffs = getattr(
                config, "phase_cutoffs", {"early": 12, "mid": 28}
            )
            self.sim_budget = getattr(
                config, "sim_budget", {"early": 300, "mid": 200, "late": 120}
            )

    def play_game(self) -> int:
        """Play a single self-play game and store the resulting data."""
        board = GomokuBoard()
        game_data = []
        recs_this_game = []
        move_number = 0
        early_entropies = []  # H(π_mcts) over legal for early plies
        open5_key = None
        tau_cutoff = getattr(self, "tau_cutoff_plies", 12)

        if hasattr(self.player1, "reset"):
            self.player1.reset()
        if hasattr(self.player2, "reset"):
            self.player2.reset()

        while not board.is_terminal():
            current_player = self.player1 if move_number % 2 == 0 else self.player2

            state_tensor = encoder.board_to_tensor(board, board.current_player)

            # Set τ schedule by move
            if hasattr(current_player, "set_temperature"):
                current_player.set_temperature(self._tau_for_move(move_number))

            # Simulation budget shaping by phase
            if hasattr(current_player.mcts, "n_simulations"):
                current_player.mcts.n_simulations = self._sims_for_move(move_number)

            root_noise = move_number == 0
            action, visit_probs = current_player.get_action(
                board, return_probs=True, root_noise=root_noise
            )

            board_size = board.board_size
            pi_arr = np.zeros((board_size, board_size), dtype=np.float32)
            for move, prob in visit_probs.items():
                r, c = move if isinstance(move, tuple) else board.index_to_move(move)
                pi_arr[r, c] = prob
            pi = torch.from_numpy(pi_arr)

            legal_mask = torch.zeros(board_size, board_size, dtype=torch.bool)
            for r, c in board.get_legal_moves():
                legal_mask[r, c] = True

            # Net policy on THIS state (optional diagnostics)
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

            state_canon = canonicalize_state(state_tensor)
            canon_hash = minhash_symmetries(state_canon)

            rec = SampleV2(
                state=state_tensor,
                pi_mcts=pi,
                v_scalar=0.0,  # fill after game ends
                legal_mask=legal_mask,
                move_number=move_number,
                sims=getattr(current_player.mcts, "n_simulations", -1),
                tau=getattr(current_player, "temperature", 1.0),
                c_puct=getattr(current_player.mcts, "c_puct", 1.5),
                dirichlet_alpha=getattr(current_player, "dirichlet_alpha", 0.3),
                dirichlet_eps=getattr(current_player, "dirichlet_epsilon", 0.25),
                entropy_pi_mcts=h_mcts,
                entropy_pi_net=h_net,
                kl_net_mcts=kl_nm,
                symmetry_id=0,
                canon_hash=canon_hash,
            )
            recs_this_game.append(rec)

            # For training buffer, remember who is to move from this state
            player_sign = 1 if move_number % 2 == 0 else -1
            game_data.append((state_tensor, pi, player_sign))

            # Apply the move and advance
            if not isinstance(action, tuple):
                action = board.index_to_move(action)
            board.apply_move(*action)

            # collect early entropies
            if move_number < tau_cutoff:
                early_entropies.append(float(h_mcts))

            # capture opening-5 key (after applying the 5th move, i.e., move_number == 4)
            # You already compute `canon_hash` for the current state BEFORE move applied,
            # so we capture open5_key on the *next* iteration or capture AFTER apply_move:
            if move_number == 4:
                # board has just applied 5th move above; recompute canonical hash here quickly
                state_after5 = encoder.board_to_tensor(board, board.current_player)
                state_canon5 = canonicalize_state(state_after5)
                open5_key = minhash_symmetries(state_canon5)

            if self.verbose:
                board.render()
            move_number += 1

        # Determine winner
        winner = board.get_winner()
        winner = -1 if winner == 2 else (0 if winner is None else 1)

        summary = {
            "type": "game_summary",
            "early_entropy_mcts_median": (
                float(np.median(early_entropies)) if early_entropies else None
            ),
            "open5_key": open5_key,
            "moves": move_number,  # final ply count
        }
        self.logger.write(summary)

        # Build final_data and write logs once with v_scalar
        final_data = []
        for (state_tensor, pi_tensor, player_sign), rec in zip(
            game_data, recs_this_game
        ):
            z = 0.0 if winner == 0 else (1.0 if winner == player_sign else -1.0)
            final_data.append((state_tensor, pi_tensor, z))
            rec.v_scalar = z

        samples = final_data  # [(state_tensor, pi_tensor, z), ...]
        metas = [
            (i, rec.move_number, rec.v_scalar) for i, rec in enumerate(recs_this_game)
        ]  # Ensure metas align with Meta type

        if self.div_manager is not None:
            accepted, accepted_metas = self.div_manager.admit_batch(samples, metas)
            accepted_ids = {i for (i, _move_no, _z) in accepted_metas}
        else:
            accepted = samples
            accepted_ids = set(range(len(recs_this_game)))

        # mark admitted precisely
        for i, rec in enumerate(recs_this_game):
            rec.admitted = 1 if i in accepted_ids else 0
            self.logger.write(rec.__dict__)

        if self.augment_fn:
            accepted = self.augment_fn(accepted)

        self.buffer.add(accepted)
        return len(accepted)

    def _phase_for_move(self, move_number: int) -> str:
        # Use config.phase_cutoffs if available; fall back to simple bands
        pc = getattr(self, "phase_cutoffs", {"early": 12, "mid": 28})
        if move_number < pc["early"]:
            return "early"
        elif move_number < pc["mid"]:
            return "mid"
        return "late"

    def _tau_for_move(self, move_number: int) -> float:
        cutoff = self.tau_cutoff_plies
        tau_early = getattr(self.config, "tau_early", 0.5) if self.config else 0.5
        return tau_early if move_number < cutoff else 0.0

    def _sims_for_move(self, move_number: int) -> int:
        budget = getattr(self, "sim_budget", {"early": 300, "mid": 200, "late": 120})
        return int(budget[self._phase_for_move(move_number)])


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
        "dirichlet_alpha": (
            "auto"
            if getattr(config, "dirichlet_alpha_mode", "auto") == "auto"
            else getattr(config, "dirichlet_alpha_fixed", 0.15)
        ),
        "dirichlet_epsilon": getattr(config, "dirichlet_epsilon", 0.25),
        "dirichlet_alpha_min": getattr(config, "dirichlet_alpha_min", 0.02),
        "dirichlet_alpha_max": getattr(config, "dirichlet_alpha_max", 0.50),
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

    diversity_manager = DiversityManager(
        DiversityManager.default_targets(window_size=config.replay_buffer_size)
    )

    runner = SelfPlayRunner(
        player1=player1,
        player2=player2,
        buffer=buffer,
        augment_fn=augment_data,
        verbose=False,
        diversity_manager=diversity_manager,
        config=config,
    )

    len_before = len(buffer)
    added_total = 0
    for i in trange(config.num_self_play_games, desc="Self-play games"):
        added_total += runner.play_game()
    len_after = len(buffer)
    print(
        f"[DEBUG] before={len_before}  added≈{added_total}  after={len_after}  delta={len_after - len_before}"
    )
    # print(f"\nBuffer filled with {len(buffer)} samples.")

    counts = diversity_manager.snapshot_counts()
    print(counts)

    if buffer_save_path:
        os.makedirs(os.path.dirname(buffer_save_path), exist_ok=True)
        buffer.save(buffer_save_path)
        print(f"Replay buffer saved to {buffer_save_path}")

    return model, buffer
