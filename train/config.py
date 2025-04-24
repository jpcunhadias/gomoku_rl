import os
from types import SimpleNamespace


def get_config():
    return SimpleNamespace(
        batch_size=16,
        learning_rate=1e-3,
        epochs=10,  # ⬅️ Increased from 1 to 10
        steps_per_epoch=20,  # You can optionally raise this
        replay_buffer_size=10000,
        save_path="checkpoints/debug_model.pth"
    )

