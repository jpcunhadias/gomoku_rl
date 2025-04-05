from types import SimpleNamespace

config = SimpleNamespace(
    batch_size=64,
    learning_rate=1e-3,
    epochs=10,
    steps_per_epoch=100,
    save_path="checkpoints/policy_value_net.pth",
    replay_buffer_size=10000
)