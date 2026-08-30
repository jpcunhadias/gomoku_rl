from types import SimpleNamespace


def get_config() -> SimpleNamespace:
    """
    Phase C Cycle 2 Configuration

    Changes from Cycle 1:
    1. Fixed tau_early_plies bug (now properly applied)
    2. Increased tau at ply 0: 0.55 → 0.85 (more stochastic exploration)
    3. Increased tau at ply 1: 0.35 → 0.50
    4. Increased tau at ply 2: 0.25 → 0.30
    5. Increased dirichlet_epsilon_root: 0.40 → 0.55 (stronger root noise)
    6. Increased dirichlet_epsilon: 0.05 → 0.12 (more exploration at plies 1-2)
    7. Increased c_puct_schedule c0: 0.3 → 0.8 (stronger exploration bonus)
    8. Decreased c_puct_schedule lambda: 0.7 → 0.5 (slower decay)
    9. Adjusted dirichlet concentration: 6.0 → 8.0
    10. Adjusted dirichlet_alpha_max: 0.08 → 0.10

    Training Adjustments (v2 - after debug check):
    11. Increased epochs: 10 → 15 (50% more training to learn sharper policies)
    12. Increased steps_per_epoch: 50 → 60 (20% more steps per epoch)

    Exploration Adjustments (v3 - after MCTS target entropy check):
    13. REDUCED tau_early_plies[0]: 0.85 → 0.70 (less stochastic at root)
    14. REDUCED tau_early_plies[1]: 0.50 → 0.42
    15. REDUCED tau_early_plies[2]: 0.30 → 0.28
    16. REDUCED dirichlet_epsilon_root: 0.55 → 0.45 (less noise at root)
    17. REDUCED dirichlet_epsilon: 0.12 → 0.08 (less noise at non-root)
    18. REDUCED c_puct_schedule c0: 0.8 → 0.6 (less aggressive exploration)

    Exploration Adjustments (v4 - after v3 results too sharp):
    19. INCREASED tau_early_plies[0]: 0.70 → 0.78 (balance between v2 and v3)
    20. INCREASED tau_early_plies[1]: 0.42 → 0.46
    21. INCREASED dirichlet_epsilon_root: 0.45 → 0.50 (moderate noise)
    22. INCREASED dirichlet_epsilon: 0.08 → 0.10 (slight increase)
    23. INCREASED c_puct_schedule c0: 0.6 → 0.65 (moderate exploration)

    Rationale:
    - v2: Normalized entropy 0.973 (too uniform), raw entropy 3.18 (good exploration)
    - v3: Normalized entropy 0.196 (too sharp), raw entropy 0.805 (too low exploration)
    - v4: Target normalized entropy 0.45-0.65 AND raw entropy >2.0
    - Solution: Intermediate values between v2 and v3 to balance both metrics
    - Goal: Good exploration (raw >2.0) AND sharp policies (normalized 0.45-0.65)

    Expected Impact:
    - Raw entropy at ply 0 should be >2.0 (good exploration)
    - Normalized entropy should be in 0.45-0.65 range (sharp but not too sharp)
    - Model will learn appropriately sharp policies from balanced data
    """
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
        epochs=15,  # INCREASED from 10 (more training to learn sharper policies)
        steps_per_epoch=60,  # INCREASED from 50 (more steps per epoch)
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
        # --- MCTS Self-Play Search Tuning (Phase C, Cycle 2) ---
        # Temperature (τ) - BALANCED for good exploration AND sharp policies (v4)
        tau_cutoff_plies=3,  # τ applies to plies 0..2
        tau_early=0.15,  # Fallback if tau_early_plies not used
        tau_early_plies={
            0: 0.78,  # BALANCED between v2 (0.85) and v3 (0.70)
            1: 0.46,  # BALANCED between v2 (0.50) and v3 (0.42)
            2: 0.28,  # Keep same as v3
        },
        # Dirichlet Noise - BALANCED for good exploration AND sharp policies (v4)
        add_dirichlet_noise=True,
        dirichlet_epsilon=0.10,  # BALANCED between v2 (0.12) and v3 (0.08)
        dirichlet_epsilon_root=0.50,  # BALANCED between v2 (0.55) and v3 (0.45)
        dirichlet_alpha_mode="auto",
        dirichlet_alpha_fixed=0.15,
        dirichlet_alpha_min=0.015,  # Slightly increased from 0.01
        dirichlet_alpha_max=0.10,  # INCREASED from 0.08
        dirichlet_concentration=8.0,  # INCREASED from 6.0
        # Simulation Budget
        sim_budget={"early": 750, "mid": 200, "late": 120},
        phase_cutoffs={"early": 12, "mid": 28},
        # C_puct Schedule - BALANCED for good exploration AND sharp policies (v4)
        c_puct_schedule={
            "enabled": True,
            "c0": 0.65,  # BALANCED between v2 (0.8) and v3 (0.6)
            "lambda_": 0.5,  # Keep same as Cycle 2
            "c_min": 1.0,
        },
        c_puct_early=0.20,
        c_puct_cutoff_plies=3,
        # --- Opening Variety (Self-Play) ---
        sp_uniform_root_p=0.0,  # Probability of forcing a random first move
        sp_block_opening_repeats=False,  # Enable opening memory guard
        opening_memory_size=300,  # Size of the recent openings cache
        restart_cap_fraction=0.20,  # Disable guard if restarts exceed this fraction
    )
