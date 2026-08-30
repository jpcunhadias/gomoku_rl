from types import SimpleNamespace


def get_config() -> SimpleNamespace:
    """
    Sweep point: clean v4 baseline (1.0x both axes), for a fair comparison basis.

    Identical tau/dirichlet to Cycle 2 (configs/phaseC_c2.py), but generated with the same
    methodology as the other sweep points (configs/sweep_tau_075.py etc.) rather than reusing
    Cycle 2's own production buffer: self-play searches with Cycle 2's trained weights, but
    starts with an EMPTY buffer (not seeded from Cycle 2's or any other buffer).

    Why this exists: Cycle 2's own buffer isn't a clean sample of v4's config alone -- 25% of
    it was seeded from Cycle 1's buffer (cold-start, untrained-network self-play, which has
    elevated entropy at every ply for unrelated reasons -- see
    docs/archive/CYCLE1_COLDSTART_MECHANISM.md). Comparing sweep points' entropy against
    Cycle 2's contaminated buffer was comparing a clean measurement to a mixed one. This config
    generates a genuinely clean 1.0x reference point instead. See
    docs/current/SWEEP_TAU_DIRICHLET.md for the full story.
    """
    return SimpleNamespace(
        seed=12345,
        cycle=50,  # placeholder; actual cycle id is always passed via --cycle / CYCLE=
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
        dirichlet_epsilon=0.10,  # v4, unscaled
        dirichlet_epsilon_root=0.50,  # v4, unscaled
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
