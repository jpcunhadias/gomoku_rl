from types import SimpleNamespace


def get_config() -> SimpleNamespace:
    return SimpleNamespace(
        num_self_play_games=100,
        self_play_num_simulations=800,
        batch_size=128,
        learning_rate=1e-3,
        epochs=10,
        steps_per_epoch=50,
        reload_buffer_every=250,
        replay_buffer_size=30000,
        save_path="checkpoints/policy_value_net.pth",
        eval_every=200,
        target_win_rate=0.9,
        eval_num_games=20,
        eval_num_simulations=800,
        # MCTS
        c_puct=1.5,
        c_puct_early=0.20,  # a touch less exploration early
        c_puct_cutoff_plies=3,  # use c_puct_early for plies 0,1,2
        c_puct_pure=2.0,
        use_rave=False,
        # Sampler / training
        use_stratified_sampler=True,
        report_sampler_mix=False,
        # --- Phase B knobs (tighter) ---
        tau_cutoff_plies=3,  # τ applies to plies 0..2 (your code uses < cutoff)
        tau_early=0.20,  # from 0.0 → 0.25
        add_dirichlet_noise=True,
        dirichlet_epsilon=0.12,  # less root noise mass
        dirichlet_alpha_mode="auto",  # α ≈ concentration / #legal (clipped)
        dirichlet_alpha_fixed=0.15,  # unused in AUTO
        dirichlet_alpha_min=0.01,  # allow smaller α
        dirichlet_alpha_max=0.03,  # tighter cap
        dirichlet_concentration=3.0,  # from 10 → 6 reduces α overall
        # simulation budget shaping
        sim_budget={
            "early": 550,
            "mid": 200,
            "late": 120,
        },  # a bit more early sims sharpens π
        # phase cutoffs (plies)
        phase_cutoffs={"early": 12, "mid": 28},
        # default MCTSPlayer temp (overridden per-move by τ schedule anyway)
        temperature=0.2,
    )
