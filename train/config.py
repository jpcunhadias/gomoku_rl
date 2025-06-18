from types import SimpleNamespace


def get_config() -> SimpleNamespace:
    return SimpleNamespace(
        num_self_play_games=50,
        self_play_num_simulations=400,
        batch_size=128,
        learning_rate=1e-3,
        epochs=1000,
        steps_per_epoch=100,
        reload_buffer_every=500,
        replay_buffer_size=10000,
        save_path="checkpoints/policy_value_net.pth",
        eval_every=100,
        target_win_rate=0.8,
        eval_num_games=20,
        eval_num_simulations=3200,
    )
