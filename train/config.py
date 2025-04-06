from types import SimpleNamespace

config = SimpleNamespace(
    # Training loop
    batch_size=128,
    learning_rate=1e-3,
    epochs=50,  # Train for 50 epochs
    steps_per_epoch=100,  # 100 batches per epoch
    # Replay buffer
    replay_buffer_size=10000,
    # Model checkpoint
    save_path="checkpoints/policy_value_net_epoch{epoch}.pth",
)
