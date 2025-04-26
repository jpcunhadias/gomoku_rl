from types import SimpleNamespace


def get_config():
    return SimpleNamespace(
        num_self_play_games=50,
        batch_size=128,
        learning_rate=1e-3,
        epochs=100,
        steps_per_epoch=100,
        replay_buffer_size=20000,
        save_path="checkpoints/policy_value_net.pth",
    )
