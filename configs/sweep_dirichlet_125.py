from types import SimpleNamespace


def get_config() -> SimpleNamespace:
    """
    Sweep point: dirichlet_epsilon/epsilon_root x1.25, tau at v4 (unscaled), c_puct fixed.
    See configs/sweep_tau_075.py for the sweep design notes.
    """
    return SimpleNamespace(
        seed=12345,
        cycle=34,
        num_self_play_games=100,
        self_play_num_simulations=800,
        reload_buffer_every=250,
        replay_buffer_size=30000,
        batch_size=128,
        learning_rate=1e-3,
        epochs=15,
        steps_per_epoch=60,
        save_path="checkpoints/policy_value_net.pth",
        use_stratified_sampler=True,
        report_sampler_mix=False,
        eval_every=200,
        target_win_rate=0.9,
        eval_num_games=20,
        eval_num_simulations=800,
        c_puct=1.5,
        c_puct_pure=2.0,
        use_rave=False,
        temperature=0.2,
        tau_cutoff_plies=3,
        tau_early=0.15,
        tau_early_plies={0: 0.78, 1: 0.46, 2: 0.28},  # v4, unscaled
        add_dirichlet_noise=True,
        dirichlet_epsilon=0.125,  # v4 x1.25
        dirichlet_epsilon_root=0.625,  # v4 x1.25
        dirichlet_alpha_mode="auto",
        dirichlet_alpha_fixed=0.15,
        dirichlet_alpha_min=0.015,
        dirichlet_alpha_max=0.10,
        dirichlet_concentration=8.0,
        sim_budget={"early": 750, "mid": 200, "late": 120},
        phase_cutoffs={"early": 12, "mid": 28},
        c_puct_schedule={"enabled": True, "c0": 0.65, "lambda_": 0.5, "c_min": 1.0},
        c_puct_early=0.20,
        c_puct_cutoff_plies=3,
        sp_uniform_root_p=0.0,
        sp_block_opening_repeats=False,
        opening_memory_size=300,
        restart_cap_fraction=0.20,
    )
