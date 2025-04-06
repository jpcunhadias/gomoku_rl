from types import SimpleNamespace

config = SimpleNamespace(
    batch_size=4,
    learning_rate=1e-3,
    epochs=1,
    steps_per_epoch=2,
    save_path="checkpoints/policy_value_net.pth",
    replay_buffer_size=20
)