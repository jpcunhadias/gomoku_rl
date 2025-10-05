# scripts/arena.py
import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
from tqdm import tqdm

from cli.play.utils import play_game
from game.gomoku import GomokuBoard
from game.player import DirichletAlphaMode, MCTSPlayer
from mcts.evaluators import NeuralEvaluator
from mcts.mcts import MCTS
from model.policy_value_net import PolicyValueNet


def load_player(ckpt, device, sims, *, use_schedule=False, schedule=None, name="MCTS"):
    model = PolicyValueNet(board_size=8).to(device)
    sd = torch.load(ckpt, map_location=device)["model_state_dict"]
    model.load_state_dict(sd)
    model.eval()
    evaluator = NeuralEvaluator(model, device)

    mcts = MCTS(
        evaluator_fn=evaluator,
        c_puct=1.5,  # base constant; schedule may override per depth
        n_simulations=sims,
        use_rave=False,
        c_puct_schedule=(schedule if use_schedule else {"enabled": False}),
    )
    player = MCTSPlayer(
        mcts,
        temperature=0.0,  # arena deterministic
        add_dirichlet_noise=False,  # arena deterministic
        dirichlet_alpha_mode=DirichletAlphaMode.AUTO,
        dirichlet_epsilon=0.0,
        name=name,
    )
    return player


def wilson_ci(p, n, z=1.96):
    denom = 1 + (z * z) / n
    center = (p + (z * z) / (2 * n)) / denom
    half = z * np.sqrt((p * (1 - p) / n) + (z * z) / (4 * n * n)) / denom
    return (center - half, center + half)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--sims", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="checkpoints/arena_result.json")
    ap.add_argument("--schedule_c0", type=float, default=1.5)
    ap.add_argument("--schedule_lambda", type=float, default=0.60)
    ap.add_argument("--schedule_cmin", type=float, default=1.0)
    ap.add_argument(
        "--baseline_schedule",
        action="store_true",
        help="If set, baseline also uses schedule (usually OFF).",
    )
    args = ap.parse_args()

    # Determinism best-effort
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    schedule = dict(
        enabled=True,
        c0=args.schedule_c0,
        lambda_=args.schedule_lambda,
        c_min=args.schedule_cmin,
    )

    base_black = load_player(
        args.baseline,
        device,
        args.sims,
        use_schedule=args.baseline_schedule,
        schedule=schedule if args.baseline_schedule else None,
        name="Baseline",
    )
    base_white = load_player(
        args.baseline,
        device,
        args.sims,
        use_schedule=args.baseline_schedule,
        schedule=schedule if args.baseline_schedule else None,
        name="Baseline",
    )
    cand_black = load_player(
        args.candidate,
        device,
        args.sims,
        use_schedule=True,
        schedule=schedule,
        name="Candidate",
    )
    cand_white = load_player(
        args.candidate,
        device,
        args.sims,
        use_schedule=True,
        schedule=schedule,
        name="Candidate",
    )

    wins = losses = draws = 0
    t0 = time.time()

    # Half games Candidate as black, half as white (color-balanced)
    half = args.games // 2
    for _ in tqdm(range(half), desc="Cand=Black vs Base=White"):
        w = play_game(GomokuBoard(), cand_black, base_white, deterministic_eval=True)
        if w == 1:
            wins += 1
        elif w == 2:
            losses += 1
        else:
            draws += 1

    for _ in tqdm(range(args.games - half), desc="Base=Black vs Cand=White"):
        w = play_game(GomokuBoard(), base_black, cand_white, deterministic_eval=True)
        if w == 1:
            losses += 1  # baseline black beat candidate white
        elif w == 2:
            wins += 1  # candidate white beat baseline black
        else:
            draws += 1

    n_decisive = wins + losses
    wr = wins / max(1, n_decisive)
    lo, hi = wilson_ci(wr, max(1, n_decisive))
    out = dict(
        games=args.games,
        decisive=n_decisive,
        wins=wins,
        losses=losses,
        draws=draws,
        winrate_decisive=wr,
        wilson95_lo=lo,
        wilson95_hi=hi,
        sims=args.sims,
        schedule=schedule,
        seed=args.seed,
        baseline=args.baseline,
        candidate=args.candidate,
        baseline_schedule=args.baseline_schedule,
        elapsed_sec=time.time() - t0,
    )
    print(json.dumps(out, indent=2))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    # Log results to CSV
    log_path = os.path.join("checkpoints", "arena", "arena_log.csv")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    write_header = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "cycle",
                "games",
                "decisive",
                "wins",
                "losses",
                "draws",
                "winrate_decisive",
                "wilson95_lo",
                "wilson95_hi",
                "sims",
                "schedule",
                "baseline",
                "candidate",
                "seed",
                "elapsed_sec",
            ],
        )
        if write_header:
            w.writeheader()
        row = {**out, "cycle": getattr(args, "cycle", -1)}
        w.writerow(row)


if __name__ == "__main__":
    main()
