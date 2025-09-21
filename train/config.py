from types import SimpleNamespace


def get_config() -> SimpleNamespace:
    return SimpleNamespace(
        num_self_play_games=50,
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
        c_puct=1.5,
        temperature=0.2,
        add_dirichlet_noise=True,
        c_puct_pure=2.0,
        use_rave=False,
        use_stratified_sampler=True,
        # --- Phase B knobs ---
        tau_cutoff_plies=4,  # τ=1 until this ply (0-indexed), then τ=0
        tau_early=0.3,
        dirichlet_epsilon=0.05,  # root noise mixture (keep)
        dirichlet_alpha_mode="auto",  # "auto" or "fixed"
        dirichlet_alpha_fixed=0.15,  # used if mode=="fixed"
        dirichlet_alpha_min=0.02,
        dirichlet_alpha_max=0.10,
        # simulation budget shaping by phase
        sim_budget=dict(early=300, mid=200, late=120),
        # (optional) phase cutoffs in plies; adjust later if you want
        phase_cutoffs=dict(early=12, mid=28),  # [0,12) early, [12,28) mid, else late
        report_sampler_mix=False,
    )
