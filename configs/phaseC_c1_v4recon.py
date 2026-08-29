from types import SimpleNamespace


def get_config() -> SimpleNamespace:
    """
    Cycle 1 — v4-seeded reconnaissance config.

    Cycle 1's own exploration knobs (tau_early_plies, dirichlet_*) produce too-uniform
    MCTS target policies (median normalized entropy ~0.96, target 0.45-0.65) — confirmed
    on a real 200-game run, with the DiversityManager draw-admission bug ruled out as the
    cause (see docs/CHANGELOG.md). Cycle 2 went through v1->v2->v3->v4 to find a balanced
    exploration setting; rather than re-deriving that from scratch for Cycle 1, this seeds
    Cycle 1's exploration params from Cycle 2's v4 (the validated "balanced" point) and cuts
    num_self_play_games way down (200 -> 40) for a fast, cheap entropy read. Not meant to
    produce a real training buffer — uses cycle=901 (scratch) to avoid overwriting the real
    Cycle 1 buffer at checkpoints/buffers/replay_c1_cycle1.pkl. Once a setting checks out,
    do one full 200-game run under the real phaseC_c1 config (cycle=1) with those values.
    """
    return SimpleNamespace(
        # --- General ---
        seed=12345,
        cycle=901,  # scratch cycle id for recon runs, not the real Cycle 1
        # --- Self-Play ---
        num_self_play_games=40,  # cut from 200 for a fast entropy read
        self_play_num_simulations=800,
        reload_buffer_every=250,
        replay_buffer_size=30000,
        # --- Training (unused for self-play-only recon runs) ---
        batch_size=128,
        learning_rate=1e-3,
        epochs=10,
        steps_per_epoch=50,
        save_path="checkpoints/policy_value_net.pth",
        use_stratified_sampler=True,
        report_sampler_mix=False,
        # --- Evaluation (unused) ---
        eval_every=200,
        target_win_rate=0.9,
        eval_num_games=20,
        eval_num_simulations=800,
        # --- MCTS Core ---
        c_puct=1.5,
        c_puct_pure=2.0,
        use_rave=False,
        temperature=0.2,
        # --- MCTS Self-Play Search Tuning (seeded from Cycle 2 v4) ---
        tau_cutoff_plies=3,
        tau_early=0.15,
        tau_early_plies={0: 0.78, 1: 0.46, 2: 0.28},  # v4
        add_dirichlet_noise=True,
        dirichlet_epsilon=0.10,  # v4
        dirichlet_epsilon_root=0.50,  # v4
        dirichlet_alpha_mode="auto",
        dirichlet_alpha_fixed=0.15,
        dirichlet_alpha_min=0.015,  # v4
        dirichlet_alpha_max=0.10,  # v4
        dirichlet_concentration=8.0,  # v4
        # Simulation Budget (unchanged from Cycle 1 / Cycle 2)
        sim_budget={"early": 750, "mid": 200, "late": 120},
        phase_cutoffs={"early": 12, "mid": 28},
        # C_puct Schedule (v4) — note: self-play forces c_puct_schedule={"enabled": False}
        # regardless (see train/self_play.py:run_selfplay_pipeline), so this only matters
        # if this config later gets reused for training.
        c_puct_schedule={"enabled": True, "c0": 0.65, "lambda_": 0.5, "c_min": 1.0},
        c_puct_early=0.20,
        c_puct_cutoff_plies=3,
        # --- Opening Variety (Self-Play) ---
        sp_uniform_root_p=0.0,
        sp_block_opening_repeats=False,
        opening_memory_size=300,
        restart_cap_fraction=0.20,
    )
