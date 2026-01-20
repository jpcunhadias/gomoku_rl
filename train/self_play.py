import json
import logging
import os
import random
import time
from collections import deque
from types import SimpleNamespace
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import torch
from tqdm import trange

from game import encoder
from game.gomoku import GomokuBoard
from game.player import DirichletAlphaMode, MCTSPlayer
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
from utils.paths import cycle_paths, save_config, save_json, save_meta

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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
        logger_path: Optional[str] = None,
    ) -> None:
        self.player1 = player1
        self.player2 = player2
        self.buffer = buffer
        self.augment_fn = augment_fn
        self.verbose = verbose
        self.logger = SampleLogger(
            logger_path or "checkpoints/selfplay/selfplay_v2.jsonl"
        )
        self.div_manager = diversity_manager
        self.config = config
        self.recent_open5_keys = deque(maxlen=200)

        # Counters for opening variety strategies
        self.games_played_in_batch = 0
        self.opening_restarts = 0
        self.uniform_root_used = 0
        self.opening_guard_disabled = False

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
        self.games_played_in_batch += 1

        # Determine if this game should use uniform root initially
        sp_uniform_root_p = getattr(self.config, "sp_uniform_root_p", 0.1)
        is_uniform_root_game_initial = random.random() < sp_uniform_root_p
        if is_uniform_root_game_initial:
            self.uniform_root_used += 1

        force_uniform_root = is_uniform_root_game_initial
        opening_memory_hit = 0

        # The restartable game loop
        while True:
            board = GomokuBoard()
            game_data = []
            recs_this_game = []
            move_number = 0
            early_entropies = []
            open5_key = None
            tau_cutoff = self.tau_cutoff_plies

            if hasattr(self.player1, "reset"):
                self.player1.reset()
            if hasattr(self.player2, "reset"):
                self.player2.reset()

            if hasattr(self.player1.mcts, "reset_root"):
                self.player1.mcts.reset_root()
            if hasattr(self.player2.mcts, "reset_root"):
                self.player2.mcts.reset_root()

            # The move loop
            while not board.is_terminal():
                current_player = self.player1 if move_number % 2 == 0 else self.player2

                state_tensor = encoder.board_to_tensor(board, board.current_player)

                # Set τ schedule by move
                if hasattr(current_player, "set_temperature"):
                    current_player.set_temperature(self._tau_for_move(move_number))

                budget = self._sims_for_move(move_number)
                if hasattr(current_player.mcts, "set_simulation_budget"):
                    current_player.mcts.set_simulation_budget(budget)
                else:
                    current_player.mcts.n_simulations = budget

                if move_number < self.tau_cutoff_plies:
                    logging.info(
                        f"[τ dbg] ply={move_number} τ={getattr(current_player, 'temperature', None)}"
                    )

                root_noise = move_number == 0

                if move_number == 0 and force_uniform_root:
                    original_temp = current_player.temperature
                    original_eps_root = current_player.dirichlet_epsilon_root

                    current_player.set_temperature(1.5)
                    current_player.dirichlet_epsilon_root = 0.8

                    action, visit_probs = current_player.get_action(
                        board, return_probs=True, root_noise=root_noise
                    )

                    current_player.set_temperature(original_temp)
                    current_player.dirichlet_epsilon_root = original_eps_root
                else:
                    action, visit_probs = current_player.get_action(
                        board, return_probs=True, root_noise=root_noise
                    )

                board_size = board.board_size
                pi_arr = np.zeros((board_size, board_size), dtype=np.float32)
                for move, prob in visit_probs.items():
                    r, c = (
                        move if isinstance(move, tuple) else board.index_to_move(move)
                    )
                    pi_arr[r, c] = prob
                pi = torch.from_numpy(pi_arr)

                legal_mask = torch.zeros(board_size, board_size, dtype=torch.bool)
                for r, c in board.get_legal_moves():
                    legal_mask[r, c] = True

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

                alpha_eff = None
                eps_eff = 0.0
                if getattr(current_player, "add_dirichlet_noise", False) and root_noise:
                    alpha_eff = current_player.get_dirichlet_alpha(board)
                    # Log the actual epsilon used (root epsilon at move 0, else regular)
                    eps_eff = (
                        current_player.dirichlet_epsilon_root
                        if move_number == 0
                        else current_player.dirichlet_epsilon
                    )

                root_cs = getattr(current_player.mcts, "last_depth_cs", [])
                rec = SampleV2(
                    state=state_tensor,
                    pi_mcts=pi,
                    v_scalar=0.0,
                    legal_mask=legal_mask,
                    move_number=move_number,
                    sims=getattr(current_player.mcts, "n_simulations", -1),
                    tau=getattr(current_player, "temperature", 1.0),
                    c_puct=float(root_cs[0])
                    if root_cs
                    else getattr(current_player.mcts, "c_puct", 1.5),
                    dirichlet_alpha=alpha_eff if alpha_eff is not None else 0.0,
                    dirichlet_eps=eps_eff,
                    entropy_pi_mcts=h_mcts,
                    entropy_pi_net=h_net,
                    kl_net_mcts=kl_nm,
                    symmetry_id=0,
                    canon_hash=canon_hash,
                )
                recs_this_game.append(rec)

                player_sign = 1 if move_number % 2 == 0 else -1
                game_data.append((state_tensor, pi, player_sign))

                if not isinstance(action, tuple):
                    action = board.index_to_move(action)
                board.apply_move(*action)

                self.player1.mcts.update_with_move(action)
                self.player2.mcts.update_with_move(action)

                if move_number < tau_cutoff:
                    early_entropies.append(float(h_mcts))

                if move_number < tau_cutoff:
                    topk = (
                        torch.topk(pi.view(-1), k=min(5, pi.numel()))
                        .values.sum()
                        .item()
                    )
                    stats = current_player.mcts.root_visit_stats() or {}
                    sched = getattr(current_player.mcts, "last_depth_cs", [])
                    root_c = (
                        f"{sched[0]:.2f}"
                        if sched and len(sched) > 0
                        else f"{current_player.mcts.c_puct:.2f}"
                    )
                    msg = (
                        f"[dbg] ply={move_number} "
                        f"τ={getattr(current_player, 'temperature', None)} "
                        f"c_root={root_c} "
                        f"Hmcts={float(h_mcts):.3f} "
                        f"top5_mass={topk:.3f}"
                    )
                    if stats:
                        msg += (
                            f" | visits n={stats.get('n_children', 0)} "
                            f"min={stats.get('min', 0)} max={stats.get('max', 0)} "
                            f"mean={stats.get('mean', 0):.1f}"
                        )
                    if sched:
                        vals = [f"{c:.2f}" for c in sched[:3]]
                        print(
                            f"[c_puct sched] ply={move_number} "
                            + " ".join(f"c@d{i}={val}" for i, val in enumerate(vals))
                        )
                    print(msg)

                if move_number == 4:
                    state_after5 = encoder.board_to_tensor(board, board.current_player)
                    state_canon5 = canonicalize_state(state_after5)
                    open5_key = minhash_symmetries(state_canon5)
                    is_repeat = open5_key in self.recent_open5_keys

                    sp_block_opening_repeats = getattr(
                        self.config, "sp_block_opening_repeats", True
                    )
                    if (
                        sp_block_opening_repeats
                        and not self.opening_guard_disabled
                        and is_repeat
                        and not force_uniform_root
                    ):
                        self.opening_restarts += 1
                        opening_memory_hit = 1
                        if (
                            self.games_played_in_batch > 10
                            and (self.opening_restarts / self.games_played_in_batch)
                            > 0.2
                        ):
                            self.opening_guard_disabled = True
                        force_uniform_root = True
                        break

                if self.verbose:
                    board.render()
                move_number += 1

            if force_uniform_root and opening_memory_hit == 1:
                opening_memory_hit = 0
                continue

            winner = board.get_winner()
            winner = -1 if winner == 2 else (0 if winner is None else 1)

            if open5_key is not None:
                self.recent_open5_keys.append(open5_key)

            summary = {
                "type": "game_summary",
                "early_entropy_mcts_median": (
                    float(np.median(early_entropies)) if early_entropies else None
                ),
                "open5_key": open5_key,
                "moves": move_number,
            }
            self.logger.write(summary)

            final_data = []
            for (state_tensor, pi_tensor, player_sign), rec in zip(
                game_data, recs_this_game
            ):
                z = 0.0 if winner == 0 else (1.0 if winner == player_sign else -1.0)
                final_data.append((state_tensor, pi_tensor, z))
                rec.v_scalar = z

            samples = final_data
            metas = [
                (i, int(rec.move_number), float(rec.v_scalar))
                for i, rec in enumerate(recs_this_game)
            ]

            if self.div_manager is not None:
                accepted, accepted_metas = self.div_manager.admit_batch(samples, metas)
                accepted_ids = {i for (i, _move_no, _z) in accepted_metas}
            else:
                accepted = samples
                accepted_ids = set(range(len(recs_this_game)))

            for i, rec in enumerate(recs_this_game):
                rec.admitted = 1 if i in accepted_ids else 0
                if i == 0:
                    rec.uniform_root_applied = 1 if is_uniform_root_game_initial else 0
                    rec.opening_memory_hit = opening_memory_hit
                rec.open5_key = open5_key
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
        """Return temperature for the given move number.

        Uses per-ply temperatures from tau_early_plies if available,
        otherwise falls back to tau_early for moves < tau_cutoff_plies.
        """
        cutoff = self.tau_cutoff_plies

        # Check for per-ply temperature mapping
        tau_early_plies = (
            getattr(self.config, "tau_early_plies", None) if self.config else None
        )

        if tau_early_plies:
            # Handle both int and str keys (JSON serialization converts int keys to str)
            if move_number in tau_early_plies:
                return float(tau_early_plies[move_number])
            elif str(move_number) in tau_early_plies:
                return float(tau_early_plies[str(move_number)])

        # Fallback to original behavior
        if move_number < cutoff:
            tau_early = getattr(self.config, "tau_early", 0.5) if self.config else 0.5
            return tau_early

        return 0.0

    def _sims_for_move(self, move_number: int) -> int:
        budget = getattr(self, "sim_budget", {"early": 300, "mid": 200, "late": 120})
        return int(budget[self._phase_for_move(move_number)])


def initialize_model(
    device: str, checkpoint_path: Optional[str] = None
) -> PolicyValueNet:
    model = PolicyValueNet(board_size=8).to(device)
    model._init_weights()
    if checkpoint_path and os.path.exists(checkpoint_path):
        logging.info(f"Loading model from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        logging.info(
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
        "dirichlet_alpha_mode": (
            DirichletAlphaMode.AUTO
            if getattr(config, "dirichlet_alpha_mode", "auto") == "auto"
            else DirichletAlphaMode.FIXED
        ),
        "dirichlet_alpha_fixed": getattr(config, "dirichlet_alpha_fixed", 0.15),
        "dirichlet_epsilon": getattr(config, "dirichlet_epsilon", 0.25),
        "dirichlet_epsilon_root": getattr(config, "dirichlet_epsilon_root", 0.30),
        "dirichlet_alpha_min": getattr(config, "dirichlet_alpha_min", 0.02),
        "dirichlet_alpha_max": getattr(config, "dirichlet_alpha_max", 0.50),
        "dirichlet_concentration": getattr(config, "dirichlet_concentration", 10.0),
    }

    mcts_kwargs = {
        "evaluator_fn": evaluator,
        "c_puct": config.c_puct,
        "n_simulations": n_simulations,
        "use_rave": config.use_rave,
        "c_puct_schedule": {"enabled": False},
    }

    player1 = MCTSPlayer(MCTS(**mcts_kwargs), **player_kwargs)
    player2 = MCTSPlayer(MCTS(**mcts_kwargs), **player_kwargs)

    return player1, player2


def run_selfplay_pipeline(
    config: Any, load_checkpoint: bool = False
) -> Tuple[PolicyValueNet, ReplayBuffer]:
    """Run a full self-play pipeline."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    cycle = int(getattr(config, "cycle", 1))
    paths = cycle_paths(cycle)
    prev_paths = cycle_paths(cycle - 1)
    
    # Ensure c_puct_schedule is disabled in saved config (self-play never uses it)
    config.c_puct_schedule = {"enabled": False}
    
    save_config(config, paths["config"])
    t0 = time.time()

    ckpt = None
    if os.path.exists(paths["model_last"]):
        ckpt = str(paths["model_last"])
    elif os.path.exists(prev_paths["model_last"]):
        ckpt = str(prev_paths["model_last"])
    elif os.path.exists(prev_paths["model_best"]):
        ckpt = str(prev_paths["model_best"])

    model = initialize_model(device, ckpt)

    evaluator = NeuralEvaluator(model, device)
    player1, player2 = create_players(
        evaluator, n_simulations=config.self_play_num_simulations, config=config
    )

    buf_path = str(paths["buffer"])
    if os.path.exists(buf_path):
        logging.info(f"Loading existing replay buffer from {buf_path}")
        buffer = ReplayBuffer.load(buf_path)
    else:
        logging.info("No existing buffer found. Checking for previous cycle buffer.")
        prev_buf_path = str(prev_paths["buffer"])
        if os.path.exists(prev_buf_path):
            logging.info(
                f"Seeding new buffer from previous cycle buffer: {prev_buf_path}"
            )
            prev = ReplayBuffer.load(prev_buf_path)

            # Seed with 25% of the buffer capacity from the tail of the previous buffer
            seed_fraction = 0.25
            seed_count = int(config.replay_buffer_size * seed_fraction)
            tail_n = min(seed_count, len(prev))

            buffer = ReplayBuffer(max_size=config.replay_buffer_size)
            buffer.add(prev.buffer[-tail_n:])

            # Shuffle the newly seeded buffer
            logging.info(f"Shuffling buffer with {len(buffer)} seeded samples.")
            buffer.shuffle()

            # Log metadata about the seeding process
            meta_path = paths["meta"]
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)
            else:
                meta = {}

            meta.update(
                {
                    "seeded_from_cycle": cycle - 1,
                    "seed_count": tail_n,
                    "seed_fraction": tail_n / config.replay_buffer_size,
                    "seed_selection": "tail_uniform",
                }
            )
            save_json(meta, meta_path)

        else:
            logging.info(
                "No previous cycle buffer found. Initializing new ReplayBuffer."
            )
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
        logger_path=str(paths["sp_log"]),
    )

    len_before = len(buffer)
    added_total = 0
    for i in trange(config.num_self_play_games, desc="Self-play games"):
        added_total += runner.play_game()
    len_after = len(buffer)
    logging.debug(
        f"[DEBUG] before={len_before}  added≈{added_total}  after={len_after}  delta={len_after - len_before}"
    )
    logging.info(f"Buffer filled with {len(buffer)} samples.")
    t1 = time.time()
    logging.info(f"Self-play of {config.num_self_play_games} games took {t1 - t0:.1f}s")

    counts = diversity_manager.snapshot_counts()
    logging.info(f"Diversity manager snapshot counts: {counts}")
    logging.debug(counts)

    buffer.save(str(paths["buffer"]))
    logging.info(f"[cycle] Replay buffer snapshot → {paths['buffer']}")

    # --- Save concise per-cycle summary ---
    sp_summary = {
        "games": config.num_self_play_games,
        "sim_budget": getattr(config, "sim_budget", None),
        "phase_cutoffs": getattr(config, "phase_cutoffs", None),
        "tau": dict(
            tau_cutoff_plies=getattr(config, "tau_cutoff_plies", None),
            tau_early=getattr(config, "tau_early", None),
        ),
        "dirichlet": dict(
            enabled=getattr(config, "add_dirichlet_noise", False),
            epsilon=getattr(config, "dirichlet_epsilon", None),
            alpha_mode=getattr(config, "dirichlet_alpha_mode", None),
            conc=getattr(config, "dirichlet_concentration", None),
            a_min=getattr(config, "dirichlet_alpha_min", None),
            a_max=getattr(config, "dirichlet_alpha_max", None),
        ),
        "c_puct_schedule": getattr(config, "c_puct_schedule", {"enabled": False}),
        "opening_variety": dict(
            uniform_root_used=runner.uniform_root_used,
            opening_restarts=runner.opening_restarts,
            opening_memory_size=len(runner.recent_open5_keys),
            uniform_root_p=getattr(config, "sp_uniform_root_p", 0.1),
            opening_guard_disabled_due_to_high_restart_rate=1
            if runner.opening_guard_disabled
            else 0,
        ),
    }

    with open(paths["sp_summary"], "w") as f:
        json.dump(sp_summary, f, indent=2)
    logging.info(f"[cycle] Self-play summary → {paths['sp_summary']}")

    # --- Save meta record ---
    meta = save_meta(
        cycle=cycle,
        seed=getattr(config, "seed", 42),
        notes="self-play run",
        extra={"elapsed_sec": time.time() - t0},
    )
    with open(paths["meta"], "w") as f:
        json.dump(meta, f, indent=2)
    logging.info(f"[cycle] Meta → {paths['meta']}")

    return model, buffer
