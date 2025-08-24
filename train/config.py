from types import SimpleNamespace


def get_config() -> SimpleNamespace:
    return SimpleNamespace(
        num_self_play_games=50,
        self_play_num_simulations=800,
        batch_size=128,
        learning_rate=1e-3,
        epochs=1000,
        steps_per_epoch=100,
        reload_buffer_every=250,
        replay_buffer_size=30000,
        save_path="checkpoints/policy_value_net.pth",
        eval_every=200,
        target_win_rate=0.9,
        eval_num_games=20,
        eval_num_simulations=800,
        c_puct=1.5,
        temperature=1.0,
        add_dirichlet_noise=True,
        c_puct_pure=2.0,
        use_rave=False,
    )
