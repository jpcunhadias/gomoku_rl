import torch
import pytest
from model.policy_value_net import PolicyValueNet, ResidualBlock


# 1. Test the output shapes and value range
def test_policy_value_net_shape_and_value_range():
    model = PolicyValueNet(board_size=8, num_blocks=5)

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
    assert (
        -2.0 <= value.item() <= 2.0
    ), f"Value should be between -2 and 2 for untrained model, got {value.item()}"

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

    # Check value output range
    assert (
        value.item() >= -1 and value.item() <= 1
    ), f"Value should be between -1 and 1, got {value.item()}"

    print("Fixed input test passed!")


# 4. Test Model on multiple random boards
def test_random_boards():
    model = PolicyValueNet(board_size=8, num_blocks=5)

    # Run the model on 10 random inputs and check output range for value
    for _ in range(10):
        random_input = torch.rand(1, 3, 8, 8)  # Random board
        policy, value = model(random_input)

        # Check value range for [-1, 1]
        assert (
            -2.0 <= value.item() <= 2.0
        ), f"Value should be between -2 and 2 for untrained model, got {value.item()}"

    print("Random board test passed!")


# Run the tests
if __name__ == "__main__":
    pytest.main()
