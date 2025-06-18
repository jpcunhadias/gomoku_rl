import torch
from types import SimpleNamespace
from train.replay_buffer import ReplayBuffer
from model.policy_value_net import PolicyValueNet
from train.train_loop import AlphaZeroTrainer


def test_train_loop_runs_without_error(tmp_path):
    # Create a small fake buffer
    buffer = ReplayBuffer(max_size=10)
    dummy_state = torch.zeros(3, 8, 8)
    dummy_pi = torch.ones(8, 8) / 64.0
    for _ in range(10):
        buffer.add([(dummy_state, dummy_pi, 2)])

    model = PolicyValueNet(board_size=8)
    config = SimpleNamespace(
        batch_size=2,
        learning_rate=1e-3,
        epochs=2,
        steps_per_epoch=2,
        save_path=str(tmp_path / "test_model.pth"),
    )

    trainer = AlphaZeroTrainer(model, buffer, config, device="cpu")
    best_epoch, best_value_loss = trainer.train()

    assert best_epoch is not None
    assert isinstance(best_value_loss, float)
