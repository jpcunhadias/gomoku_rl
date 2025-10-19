from types import SimpleNamespace


def get_config() -> SimpleNamespace:
    return SimpleNamespace(
        # --- General ---
        seed=12345,
        cycle=2,
        # --- Self-Play ---
        num_self_play_games=200,
        self_play_num_simulations=800,
        reload_buffer_every=250,
        replay_buffer_size=30000,
        # --- Training ---
        batch_size=128,
        learning_rate=1e-3,
        epochs=30,
        steps_per_epoch=50,
        save_path="checkpoints/policy_value_net.pth",
        use_stratified_sampler=True,
        report_sampler_mix=False,
        # --- Evaluation ---
        eval_every=200,
        target_win_rate=0.9,
        eval_num_games=20,
        eval_num_simulations=800,
        # --- MCTS Core ---
        c_puct=1.5,
        c_puct_pure=2.0,
        use_rave=False,
        temperature=0.2,  # Default MCTSPlayer temp (overridden by τ schedule)
        # --- MCTS Self-Play Search Tuning (Phase C) ---
        # Temperature (τ)
        tau_cutoff_plies=3,  # τ applies to plies 0..2
        tau_early=0.15,  # Fallback if tau_early_plies not used
        tau_early_plies={0: 0.55, 1: 0.35, 2: 0.25},
        # Dirichlet Noise
        add_dirichlet_noise=True,
        dirichlet_epsilon=0.03,  # Non-root noise
        dirichlet_epsilon_root=0.40,  # Root-only noise
        dirichlet_alpha_mode="auto",
        dirichlet_alpha_fixed=0.15,
        dirichlet_alpha_min=0.01,
        dirichlet_alpha_max=0.06,
        dirichlet_concentration=5.0,
        # Simulation Budget
        sim_budget={"early": 750, "mid": 200, "late": 120},
        phase_cutoffs={"early": 12, "mid": 28},
        # C_puct Schedule
        c_puct_schedule=dict(enabled=True, c0=3.0, lambda_=0.25, c_min=1.0),
        c_puct_early=0.20,
        c_puct_cutoff_plies=3,
        # --- Opening Variety (Self-Play) ---
        sp_uniform_root_p=0.15,  # Probability of forcing a random first move
        sp_block_opening_repeats=True,  # Enable opening memory guard
        opening_memory_size=300,  # Size of the recent openings cache
        restart_cap_fraction=0.20,  # Disable guard if restarts exceed this fraction
    )
