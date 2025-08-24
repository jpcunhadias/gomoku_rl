import torch
from types import SimpleNamespace
from train.replay_buffer import ReplayBuffer
from model.policy_value_net import PolicyValueNet
from train.train_loop import AlphaZeroTrainer


def test_train_loop_runs_without_error(tmp_path):
    # Create a small fake buffer with targets: -1 (loss), 0 (draw), 1 (win)
    buffer = ReplayBuffer(max_size=10)
    dummy_state = torch.zeros(3, 8, 8)
    dummy_pi = torch.ones(8, 8) / 64.0
    outcomes = [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0]
    for z in outcomes:
        buffer.add([(dummy_state.clone(), dummy_pi.clone(), z)])

    model = PolicyValueNet(board_size=8)
    model._init_weights()
    config = SimpleNamespace(
        batch_size=2,
        learning_rate=1e-3,
        epochs=2,
        steps_per_epoch=2,
        save_path=str(tmp_path / "test_model.pth"),
    )

    value_params = list(model.value_conv.parameters()) + list(
        model.value_fc.parameters()
    )
    value_param_ids = {id(p) for p in value_params}
    policy_params = [p for p in model.parameters() if id(p) not in value_param_ids]

    optimizer = torch.optim.Adam(
        [
            {"params": policy_params, "lr": config.learning_rate},
            {
                "params": value_params,
                "lr": config.learning_rate * 0.3,
                "weight_decay": 2e-4,
            },
        ]
    )

    trainer = AlphaZeroTrainer(
        model=model,
        optimizer=optimizer,
        replay_buffer=buffer,
        config=config,
        device="cpu",
    )
    best_epoch, best_value_loss = trainer.train()

    assert best_epoch is not None
    assert isinstance(best_value_loss, float)
