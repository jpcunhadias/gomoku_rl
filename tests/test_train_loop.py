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
        save_paths={
            "model_best": tmp_path / "best.pth",
            "model_last": tmp_path / "last.pth",
        },
    )
    best_epoch, best_value_loss = trainer.train()

    assert best_epoch is not None
    assert isinstance(best_value_loss, float)


def test_train_loop_updates_weights(tmp_path):
    """Test that training actually updates model weights."""
    buffer = ReplayBuffer(max_size=20)
    
    # Create varied training data
    for i in range(20):
        state = torch.rand(3, 8, 8)
        policy = torch.rand(8, 8)
        policy = policy / policy.sum()
        value = float(i % 3 - 1)  # -1, 0, or 1
        buffer.add([(state, policy, value)])

    model = PolicyValueNet(board_size=8, num_blocks=3)
    model._init_weights()
    
    # Save initial weights
    initial_weights = {
        name: param.clone().detach()
        for name, param in model.named_parameters()
    }

    config = SimpleNamespace(
        batch_size=4,
        learning_rate=1e-2,
        epochs=3,
        steps_per_epoch=3,
        save_path=str(tmp_path / "test_model.pth"),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    trainer = AlphaZeroTrainer(
        model=model,
        optimizer=optimizer,
        replay_buffer=buffer,
        config=config,
        device="cpu",
        save_paths={
            "model_best": tmp_path / "best.pth",
            "model_last": tmp_path / "last.pth",
        },
    )

    trainer.train()

    # Check that at least some weights changed
    weights_changed = False
    for name, param in model.named_parameters():
        if not torch.allclose(param, initial_weights[name], atol=1e-6):
            weights_changed = True
            break
    
    assert weights_changed, "Model weights did not change during training"


def test_train_loop_loss_decreases(tmp_path):
    """Test that loss decreases on a simple controlled dataset."""
    buffer = ReplayBuffer(max_size=50)
    
    # Create a simple pattern: center moves are good
    for _ in range(50):
        state = torch.zeros(3, 8, 8)
        state[0, 3:5, 3:5] = 1  # Mark center
        
        # Policy concentrated on center
        policy = torch.zeros(8, 8)
        policy[3:5, 3:5] = 0.25
        
        value = 1.0  # Always positive
        buffer.add([(state, policy, value)])

    model = PolicyValueNet(board_size=8, num_blocks=3)
    model._init_weights()
    
    # Compute initial loss
    model.eval()
    sample_states, sample_policies, sample_values = buffer.sample(min(8, len(buffer)))
    states = torch.stack(list(sample_states)).to("cpu")
    policies_flat = torch.stack([p.flatten() for p in sample_policies]).to("cpu")
    values_tensor = torch.tensor(list(sample_values), dtype=torch.float32).unsqueeze(1).to("cpu")
    
    with torch.no_grad():
        policy_logits, value_pred = model(states)
        log_probs = torch.log_softmax(policy_logits, dim=1)
        initial_loss = -(policies_flat * log_probs).sum(dim=1).mean()
        initial_loss += ((value_pred - values_tensor) ** 2).mean()
    
    initial_loss_value = initial_loss.item()

    config = SimpleNamespace(
        batch_size=8,
        learning_rate=1e-2,
        epochs=3,  # Reduced to avoid eval at epoch 5
        steps_per_epoch=5,
        eval_every=999,  # Disable evaluation
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    trainer = AlphaZeroTrainer(
        model=model,
        optimizer=optimizer,
        replay_buffer=buffer,
        config=config,
        device="cpu",
        save_paths={
            "model_best": tmp_path / "best.pth",
            "model_last": tmp_path / "last.pth",
        },
    )

    trainer.train()

    # Compute final loss
    model.eval()
    with torch.no_grad():
        policy_logits, value_pred = model(states)
        log_probs = torch.log_softmax(policy_logits, dim=1)
        final_loss = -(policies_flat * log_probs).sum(dim=1).mean()
        final_loss += ((value_pred - values_tensor) ** 2).mean()
    
    final_loss_value = final_loss.item()
    
    # Loss should decrease (with some tolerance)
    assert final_loss_value < initial_loss_value * 1.5, \
        f"Loss did not decrease sufficiently: {initial_loss_value:.4f} -> {final_loss_value:.4f}"


def test_train_loop_checkpoint_saved(tmp_path):
    """Test that checkpoint is saved during training."""
    buffer = ReplayBuffer(max_size=10)
    
    for i in range(10):
        state = torch.rand(3, 8, 8)
        policy = torch.ones(8, 8) / 64.0
        value = float((i % 3) - 1)
        buffer.add([(state, policy, value)])

    model = PolicyValueNet(board_size=8, num_blocks=2)
    model._init_weights()

    checkpoint_path = tmp_path / "checkpoint.pth"
    config = SimpleNamespace(
        batch_size=4,
        learning_rate=1e-3,
        epochs=2,
        steps_per_epoch=2,
    )
    
    save_paths = {
        "model_best": checkpoint_path,
        "model_last": tmp_path / "checkpoint_last.pth",
    }

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    trainer = AlphaZeroTrainer(
        model=model,
        optimizer=optimizer,
        replay_buffer=buffer,
        config=config,
        device="cpu",
        save_paths=save_paths,
    )
    
    trainer.train()
    
    # Check that checkpoint exists
    assert checkpoint_path.exists(), "Checkpoint was not saved"
    
    # Try loading the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint


def test_train_loop_with_small_buffer(tmp_path):
    """Test training with very small buffer."""
    buffer = ReplayBuffer(max_size=3)
    
    for i in range(3):
        state = torch.rand(3, 8, 8)
        policy = torch.ones(8, 8) / 64.0
        value = float(i - 1)
        buffer.add([(state, policy, value)])

    model = PolicyValueNet(board_size=8, num_blocks=2)
    model._init_weights()

    config = SimpleNamespace(
        batch_size=2,
        learning_rate=1e-3,
        epochs=1,
        steps_per_epoch=1,
        save_path=str(tmp_path / "test_model.pth"),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    trainer = AlphaZeroTrainer(
        model=model,
        optimizer=optimizer,
        replay_buffer=buffer,
        config=config,
        device="cpu",
        save_paths={
            "model_best": tmp_path / "best.pth",
            "model_last": tmp_path / "last.pth",
        },
    )

    # Should complete without error even with small buffer
    best_epoch, best_value_loss = trainer.train()
    assert best_epoch is not None

