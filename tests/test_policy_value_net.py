import torch
import pytest
from model.policy_value_net import PolicyValueNet, ResidualBlock


# 1. Test the output shapes and value range
def test_policy_value_net_shape_and_value_range():
    model = PolicyValueNet(board_size=8, num_blocks=5)
    model._init_weights()

    # Test forward pass with random input (1, 3, 8, 8)
    dummy_input = torch.rand(1, 3, 8, 8)  # Batch size of 1
    policy, value = model(dummy_input)

    # Check the shapes
    assert policy.shape == (
        1,
        64,
    ), f"Expected policy shape [1, 64], got {policy.shape}"
    assert value.shape == (1, 1), f"Expected value shape [1, 1], got {value.shape}"

    # Check value range for [-1, 1]
    expected_value = value.item()
    assert -1.0 <= expected_value <= 1.0

    print("Shape and value range test passed!")


# 2. Test Residual Block
def test_residual_block():
    block = ResidualBlock(64, 64)
    x = torch.rand(1, 64, 8, 8)  # random input tensor
    out = block(x)
    assert out.shape == x.shape, f"Expected output shape {x.shape}, but got {out.shape}"
    print("Residual block test passed!")


# 3. Test Forward pass with a known fixed pattern
def test_fixed_input():
    model = PolicyValueNet(board_size=8, num_blocks=5)
    model._init_weights()

    # Create a fixed board: Player 1's pieces (1) and Player 2's pieces (2) in known locations
    fixed_board = torch.zeros(1, 3, 8, 8)
    fixed_board[0, 0, 0, 0] = 1  # Player 1's piece at top-left
    fixed_board[0, 1, 7, 7] = 1  # Player 2's piece at bottom-right

    # Forward pass through the network
    policy, value = model(fixed_board)

    # Check shapes
    assert policy.shape == (
        1,
        64,
    ), f"Expected policy shape [1, 64], got {policy.shape}"
    assert value.shape == (1, 1), f"Expected value shape [1, 1], got {value.shape}"

    expected_value = value.item()
    assert -1.0 <= expected_value <= 1.0

    print("Fixed input test passed!")


# 4. Test Model on multiple random boards
def test_random_boards():
    model = PolicyValueNet(board_size=8, num_blocks=5)
    model._init_weights()

    # Run the model on 10 random inputs and check output range for value
    for _ in range(10):
        random_input = torch.rand(1, 3, 8, 8)  # Random board
        policy, value = model(random_input)

        expected_value = value.item()
        assert -1.0 <= expected_value <= 1.0

    print("Random board test passed!")


# 5. Test batch processing
def test_batch_processing():
    model = PolicyValueNet(board_size=8, num_blocks=5)
    model._init_weights()
    
    batch_size = 4
    batch_input = torch.rand(batch_size, 3, 8, 8)
    policy, value = model(batch_input)
    
    assert policy.shape == (batch_size, 64)
    assert value.shape == (batch_size, 1)
    
    for i in range(batch_size):
        assert -1.0 <= value[i].item() <= 1.0


# 6. Test single training step (forward + backward pass)
def test_training_step():
    model = PolicyValueNet(board_size=8, num_blocks=5)
    model._init_weights()
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create dummy batch
    batch_input = torch.rand(4, 3, 8, 8)
    target_policy = torch.rand(4, 64)
    target_policy = target_policy / target_policy.sum(dim=1, keepdim=True)  # Normalize
    target_value = torch.rand(4, 1) * 2 - 1  # Range [-1, 1]
    
    # Forward pass
    pred_policy, pred_value = model(batch_input)
    
    # Compute losses
    policy_loss = -(target_policy * torch.log_softmax(pred_policy, dim=1)).sum(dim=1).mean()
    value_loss = ((pred_value - target_value) ** 2).mean()
    total_loss = policy_loss + value_loss
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    # Check that loss is finite
    assert torch.isfinite(total_loss)
    assert total_loss.item() > 0


# 7. Test gradient flow
def test_gradient_flow():
    model = PolicyValueNet(board_size=8, num_blocks=5)
    model._init_weights()
    model.train()
    
    batch_input = torch.rand(2, 3, 8, 8)
    target_policy = torch.rand(2, 64)
    target_policy = target_policy / target_policy.sum(dim=1, keepdim=True)
    target_value = torch.ones(2, 1)
    
    pred_policy, pred_value = model(batch_input)
    
    # Compute loss for both heads
    policy_loss = -(target_policy * torch.log_softmax(pred_policy, dim=1)).sum(dim=1).mean()
    value_loss = ((pred_value - target_value) ** 2).mean()
    total_loss = policy_loss + value_loss
    
    total_loss.backward()
    
    # Check that gradients exist for all parameters
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"


# Run the tests
if __name__ == "__main__":
    pytest.main()
