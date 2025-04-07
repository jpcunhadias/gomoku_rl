import os
from types import SimpleNamespace


def get_config():
    return SimpleNamespace(
        batch_size=128,
        learning_rate=1e-3,
        epochs=50,
        steps_per_epoch=100,
        replay_buffer_size=10000,
        save_path=os.getenv(
            "MODEL_SAVE_PATH",
            "/content/drive/MyDrive/gomoku_checkpoints/policy_value_net.pth",
        ),
    )
