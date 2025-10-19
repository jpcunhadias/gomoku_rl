# scripts/arena.py
import argparse
import csv
import json
import os
from utils.seeding import set_global_seed


def load_player(
    ckpt,
    device,
    sims,
    use_schedule=False,
    schedule=None,
    stochastic=False,
    root_eps=0.0,
    tau0=0.0,
    tau1=0.0,
    name="MCTS",
):
    """
    Load a player from checkpoint with optional scheduler and stochastic evaluation.

    Args:
        ckpt: Path to checkpoint file
        device: Device to load model on
        sims: Number of MCTS simulations
        use_schedule: Whether to use c_puct scheduler
        schedule: Schedule dict with c0, lambda_, c_min
        stochastic: Whether to enable stochastic evaluation
        root_eps: Dirichlet epsilon at root (ply 0)
        tau0: Temperature at ply 0
        tau1: Temperature at ply 1
        name: Player name
    """
    # Build PolicyValueNet, load model_state_dict, eval()
    model = PolicyValueNet(board_size=8).to(device)
    
    # Load checkpoint with error handling
    try:
        sd = torch.load(ckpt, map_location=device)["model_state_dict"]
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint {ckpt}: {e}")
    
    model.load_state_dict(sd)
    model.eval()

    # Construct NeuralEvaluator, then MCTS
    evaluator = NeuralEvaluator(model, device)
    mcts = MCTS(
        evaluator_fn=evaluator,
        c_puct=1.5,  # base constant; schedule may override per depth
        n_simulations=sims,
        use_rave=False,
        c_puct_schedule=(schedule if use_schedule else {"enabled": False}),
    )

    # Wrap with MCTSPlayer (deterministic by default for arena)
    player = MCTSPlayer(
        mcts,
        temperature=0.0,
        add_dirichlet_noise=False,
        dirichlet_alpha_mode=DirichletAlphaMode.AUTO,
        dirichlet_epsilon=0.0,
        name=name,
    )

    # If stochastic=True, store arena-specific parameters on the player object
    if stochastic:
        player.set_arena_params(root_eps, tau0, tau1)

    return player


def wilson_ci(p, n, z=1.96):
    denom = 1 + (z * z) / n
    center = (p + (z * z) / (2 * n)) / denom
    half = z * np.sqrt((p * (1 - p) / n) + (z * z) / (4 * n * n)) / denom
    return (center - half, center + half)


def _normalize_opening(op):
    out = []
    for mv in op:
        if isinstance(mv, (list, tuple)) and len(mv) == 2:
            r, c = int(mv[0]), int(mv[1])
        elif isinstance(mv, dict) and "r" in mv and "c" in mv:
            r, c = int(mv["r"]), int(mv["c"])
        else:
            raise ValueError(f"Bad opening move format: {mv}")
        out.append((r, c))
    return out


def play_one(p_black, p_white, opening=None, stochastic=False):
    """
    Play a single game between p_black and p_white with optional opening and stochastic eval.

    Args:
        p_black: Black player (MCTSPlayer)
        p_white: White player (MCTSPlayer)
        opening: Optional list of (row, col) moves to apply before starting
        stochastic: If True, apply stochastic evaluation (temperature and root noise)

    Returns:
        Winner: 1 (black wins), 2 (white wins), or 0 (draw)
    """
    board = GomokuBoard()

    # Reset players & roots
    for p in (p_black, p_white):
        if hasattr(p, "reset"):
            p.reset()
        if hasattr(p, "mcts") and hasattr(p.mcts, "reset_root"):
            p.mcts.reset_root()

    # Make sure temps are clean at game start
    # (Don't persist stochastic temps across games)
    for p in (p_black, p_white):
        if hasattr(p, "set_temperature"):
            p.set_temperature(0.0)

    # Apply opening (list of (r,c)), alternating players
    if opening:
        opening = _normalize_opening(opening)
        for r, c in opening:
            board.apply_move(r, c)
            # Update both trees
            if hasattr(p_black.mcts, "update_with_move"):
                p_black.mcts.update_with_move((r, c))
            if hasattr(p_white.mcts, "update_with_move"):
                p_white.mcts.update_with_move((r, c))

    move_no = 0
    while not board.is_terminal():
        p = p_black if board.current_player == 1 else p_white

        # Stochastic eval: small τ in first plies and root-only Dirichlet one time
        root_noise = False
        original_dirichlet = None
        original_epsilon = None

        if stochastic:
            if move_no == 0:
                if hasattr(p, "set_temperature"):
                    p.set_temperature(getattr(p, "_arena_tau0", 0.0))
                root_noise = True
                # Temporarily enable Dirichlet noise for root (with safety guards)
                if getattr(p, "_arena_root_eps", 0.0) > 0.0:
                    original_dirichlet = getattr(p, "add_dirichlet_noise", None)
                    original_epsilon = getattr(p, "dirichlet_epsilon", None)
                    if original_dirichlet is not None:
                        p.add_dirichlet_noise = True
                    if original_epsilon is not None:
                        p.dirichlet_epsilon = p._arena_root_eps
            elif move_no == 1:
                if hasattr(p, "set_temperature"):
                    p.set_temperature(getattr(p, "_arena_tau1", 0.0))
            else:
                if hasattr(p, "set_temperature"):
                    p.set_temperature(0.0)

        action = p.get_action(
            board,
            return_probs=False,
            root_noise=root_noise,
        )

        # Restore original Dirichlet settings after root move (with safety guards)
        if stochastic and move_no == 0 and root_noise:
            if original_dirichlet is not None:
                p.add_dirichlet_noise = original_dirichlet
            if original_epsilon is not None:
                p.dirichlet_epsilon = original_epsilon

        if not isinstance(action, tuple):
            action = board.index_to_move(action)

        board.apply_move(*action)

        if hasattr(p_black.mcts, "update_with_move"):
            p_black.mcts.update_with_move(action)
        if hasattr(p_white.mcts, "update_with_move"):
            p_white.mcts.update_with_move(action)

        move_no += 1

    w = board.get_winner()  # 1=Black, 2=White, None=draw
    return w if w is not None else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cycle", type=int, default=-1, help="Experiment cycle id for logging"
    )
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument(
        "--games",
        type=int,
        default=200,
        help="Number of games (must be even for paired matches)",
    )
    ap.add_argument("--sims", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="checkpoints/arena/arena_result.json")
    ap.add_argument("--csv", type=str, default="checkpoints/arena/arena_log.csv")
    ap.add_argument("--schedule_c0", type=float, default=2.5)
    ap.add_argument("--schedule_lambda", type=float, default=0.30)
    ap.add_argument("--schedule_cmin", type=float, default=1.0)
    ap.add_argument(
        "--baseline_schedule",
        action="store_true",
        help="If set, baseline also uses schedule (usually OFF).",
    )
    ap.add_argument(
        "--stochastic_eval",
        action="store_true",
        default=False,
        help="Enable stochastic evaluation with Dirichlet noise and temperature",
    )
    ap.add_argument(
        "--eval_root_eps",
        type=float,
        default=0.12,
        help="Dirichlet epsilon at root (ply 0)",
    )
    ap.add_argument(
        "--eval_tau0", type=float, default=0.08, help="Temperature at ply 0"
    )
    ap.add_argument(
        "--eval_tau1", type=float, default=0.05, help="Temperature at ply 1"
    )
    ap.add_argument(
        "--opening_set",
        type=str,
        default=None,
        help="Path to JSON/JSONL file with opening positions",
    )
    args = ap.parse_args()

    # Validate that games is even for paired matches
    if args.games % 2 != 0:
        raise ValueError(f"--games must be even (got {args.games})")

    # Determinism: seed all random sources
    set_global_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    schedule = dict(
        enabled=True,
        c0=args.schedule_c0,
        lambda_=args.schedule_lambda,
        c_min=args.schedule_cmin,
    )

    # Load opening set if provided
    openings = None
    if args.opening_set:
        with open(args.opening_set, "r") as f:
            content = f.read()
            # Try JSON first, then JSONL
            try:
                openings = json.loads(content)
            except json.JSONDecodeError:
                openings = [
                    json.loads(line)
                    for line in content.strip().split("\n")
                    if line.strip()
                ]
        print(f"Loaded {len(openings)} openings from {args.opening_set}")

        # Shuffle opening set with local RNG seeded from --seed for reproducibility
        opening_rng = random.Random(args.seed)
        opening_rng.shuffle(openings)
        print(f"Shuffled openings using seed {args.seed}")

    # Build four players: baseline (no schedule by default) and candidate (with schedule)
    base_black = load_player(
        args.baseline,
        device,
        args.sims,
        use_schedule=args.baseline_schedule,
        schedule=schedule if args.baseline_schedule else None,
        stochastic=args.stochastic_eval,
        root_eps=args.eval_root_eps,
        tau0=args.eval_tau0,
        tau1=args.eval_tau1,
        name="Baseline",
    )
    base_white = load_player(
        args.baseline,
        device,
        args.sims,
        use_schedule=args.baseline_schedule,
        schedule=schedule if args.baseline_schedule else None,
        stochastic=args.stochastic_eval,
        root_eps=args.eval_root_eps,
        tau0=args.eval_tau0,
        tau1=args.eval_tau1,
        name="Baseline",
    )
    cand_black = load_player(
        args.candidate,
        device,
        args.sims,
        use_schedule=True,
        schedule=schedule,
        stochastic=args.stochastic_eval,
        root_eps=args.eval_root_eps,
        tau0=args.eval_tau0,
        tau1=args.eval_tau1,
        name="Candidate",
    )
    cand_white = load_player(
        args.candidate,
        device,
        args.sims,
        use_schedule=True,
        schedule=schedule,
        stochastic=args.stochastic_eval,
        root_eps=args.eval_root_eps,
        tau0=args.eval_tau0,
        tau1=args.eval_tau1,
        name="Candidate",
    )

    # Track detailed per-color outcomes
    cand_black_wins = 0
    cand_white_wins = 0
    base_black_wins = 0
    base_white_wins = 0
    draws = 0
    t0 = time.time()

    # Determine pairs: either openings or dummy pairs for standard games
    if openings:
        # Each opening is played twice (color-swapped)
        pairs = openings
        desc_prefix = "Opening"
    else:
        # Standard paired games without openings
        pairs = [None] * (args.games // 2)
        desc_prefix = "Pair"

    # Play paired games
    for opening in tqdm(
        pairs, desc=f"Playing {len(pairs)} {desc_prefix.lower()} pairs"
    ):
        # Game A: Candidate as black vs Baseline as white
        w_a = play_one(
            cand_black, base_white, opening=opening, stochastic=args.stochastic_eval
        )
        if w_a == 1:
            cand_black_wins += 1
        elif w_a == 2:
            base_white_wins += 1
        else:
            draws += 1

        # Game B: Baseline as black vs Candidate as white (same opening, color-swapped)
        w_b = play_one(
            base_black, cand_white, opening=opening, stochastic=args.stochastic_eval
        )
        if w_b == 1:
            base_black_wins += 1
        elif w_b == 2:
            cand_white_wins += 1
        else:
            draws += 1

    # Aggregate from candidate perspective
    wins = cand_black_wins + cand_white_wins
    losses = base_black_wins + base_white_wins
    total_games = wins + losses + draws

    # Compute decisive games and winrate
    n_decisive = wins + losses
    wr = wins / max(1, n_decisive)
    lo, hi = wilson_ci(wr, max(1, n_decisive))

    # Build comprehensive output with all required fields
    out = dict(
        # Game statistics
        games=total_games,
        decisive=n_decisive,
        wins=wins,
        losses=losses,
        draws=draws,
        winrate_decisive=wr,
        wilson95_lo=lo,
        wilson95_hi=hi,
        # Per-color breakdown
        cand_black_wins=cand_black_wins,
        cand_white_wins=cand_white_wins,
        base_black_wins=base_black_wins,
        base_white_wins=base_white_wins,
        # MCTS configuration
        sims=args.sims,
        schedule=schedule,
        seed=args.seed,
        # Model paths
        baseline=args.baseline,
        candidate=args.candidate,
        # Schedule flags
        baseline_schedule=args.baseline_schedule,
        candidate_schedule=True,  # candidate always uses schedule
        # Stochastic evaluation settings
        stochastic_eval=args.stochastic_eval,
        eval_root_eps=args.eval_root_eps if args.stochastic_eval else None,
        eval_tau0=args.eval_tau0 if args.stochastic_eval else None,
        eval_tau1=args.eval_tau1 if args.stochastic_eval else None,
        # Opening set info
        opening_set=args.opening_set,
        opening_count=len(openings) if openings else None,
        # Timing
        elapsed_sec=time.time() - t0,
    )

    # Save JSON result
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    # Log results to CSV with fixed fieldnames
    log_path = args.csv
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    write_header = not os.path.exists(log_path)

    # Fixed fieldnames that match all output dict keys
    # This ensures future runs won't fail due to missing fields
    fieldnames = [
        "cycle",
        "games",
        "decisive",
        "wins",
        "losses",
        "draws",
        "winrate_decisive",
        "wilson95_lo",
        "wilson95_hi",
        "cand_black_wins",
        "cand_white_wins",
        "base_black_wins",
        "base_white_wins",
        "sims",
        "schedule",
        "seed",
        "baseline",
        "candidate",
        "baseline_schedule",
        "candidate_schedule",
        "stochastic_eval",
        "eval_root_eps",
        "eval_tau0",
        "eval_tau1",
        "opening_set",
        "opening_count",
        "elapsed_sec",
    ]

    with open(log_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        row = {**out, "cycle": getattr(args, "cycle", -1)}
        w.writerow(row)

    # ========================================================================
    # ACCEPTANCE CHECKS & SUMMARY
    # ========================================================================

    print("\n" + "=" * 80)
    print("ARENA EVALUATION SUMMARY")
    print("=" * 80)

    # Assertion: games must be even (paired requirement)
    assert total_games % 2 == 0, (
        f"Games must be even for paired evaluation (got {total_games})"
    )
    print(
        f"✓ Paired games requirement satisfied: {total_games} total games ({len(pairs)} pairs)"
    )

    print("\n" + "-" * 80)
    print("GAME RESULTS")
    print("-" * 80)
    print(f"  Total games:     {total_games}")
    print(f"  Decisive games:  {n_decisive}")
    print(f"  Draws:           {draws}")
    print(f"  Candidate wins:  {wins}")
    print(f"  Baseline wins:   {losses}")
    print(f"  Winrate (decisive): {wr:.4f} ({wr * 100:.2f}%)")
    print(f"  Wilson 95% CI:   [{lo:.4f}, {hi:.4f}]")

    print("\n" + "-" * 80)
    print("PER-COLOR BREAKDOWN")
    print("-" * 80)
    print(f"  Candidate as Black: {cand_black_wins} wins")
    print(f"  Candidate as White: {cand_white_wins} wins")
    print(f"  Baseline as Black:  {base_black_wins} wins")
    print(f"  Baseline as White:  {base_white_wins} wins")

    # Color balance check
    cand_total = cand_black_wins + cand_white_wins
    base_total = base_black_wins + base_white_wins
    print(f"\n  Total: Candidate {cand_total} | Baseline {base_total} | Draws {draws}")

    # Per-color winrates for candidate
    games_per_color = len(pairs)
    if games_per_color > 0:
        cand_black_wr = cand_black_wins / games_per_color
        cand_white_wr = cand_white_wins / games_per_color
        print(f"  Candidate Black WR: {cand_black_wr:.4f} ({cand_black_wr * 100:.2f}%)")
        print(f"  Candidate White WR: {cand_white_wr:.4f} ({cand_white_wr * 100:.2f}%)")

    print("\n" + "-" * 80)
    print("CONFIGURATION")
    print("-" * 80)
    print(f"  Baseline:        {args.baseline}")
    print(f"  Candidate:       {args.candidate}")
    print(f"  Simulations:     {args.sims}")
    print(f"  Seed:            {args.seed}")
    print(
        f"  Schedule (c0, λ, cmin): ({args.schedule_c0}, {args.schedule_lambda}, {args.schedule_cmin})"
    )
    print(f"  Baseline uses schedule: {args.baseline_schedule}")
    print("  Candidate uses schedule: True (always)")

    if args.stochastic_eval:
        print("\n  Stochastic Evaluation: ENABLED")
        print(f"    Root epsilon (ply 0): {args.eval_root_eps}")
        print(f"    Temperature (ply 0):  {args.eval_tau0}")
        print(f"    Temperature (ply 1):  {args.eval_tau1}")
    else:
        print("\n  Stochastic Evaluation: DISABLED (deterministic)")

    if args.opening_set:
        print(f"\n  Opening set:     {args.opening_set}")
        print(f"  Openings loaded: {len(openings) if openings is not None else 0}")
    else:
        print("\n  Opening set:     None (playing from empty board)")

    print("\n" + "-" * 80)
    print("OUTPUT")
    print("-" * 80)
    print(f"  JSON result: {args.out}")
    print(f"  CSV log:     {args.csv}")
    print(f"  Elapsed:     {out['elapsed_sec']:.2f} seconds")

    print("\n" + "=" * 80)
    print("ARENA EVALUATION COMPLETE")
    print("=" * 80)

    # Print JSON summary to stdout
    print("\nJSON Summary:")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
