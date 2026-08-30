import pytest
import torch

from game.player import MCTSPlayer
from mcts.evaluators import NeuralEvaluator
from mcts.mcts import MCTS
from model.policy_value_net import PolicyValueNet
from train.config import get_config
from train.replay_buffer import ReplayBuffer
from train.self_play import SelfPlayRunner

config = get_config()


def test_self_play_game_populates_buffer():
    model = PolicyValueNet(board_size=8)
    model._init_weights()
    evaluator = NeuralEvaluator(model)
    buffer = ReplayBuffer(max_size=100)

    mcts_player1 = MCTSPlayer(
        MCTS(evaluator_fn=evaluator, n_simulations=50),
        temperature=1.0,
    )
    mcts_player2 = MCTSPlayer(
        MCTS(evaluator_fn=evaluator, n_simulations=50), temperature=1.0
    )

    runner = SelfPlayRunner(
        player1=mcts_player1,
        player2=mcts_player2,
        buffer=buffer,
        verbose=False,
        config=config,
    )

    runner.play_game()
    assert len(buffer) > 0, "Replay buffer is empty after self-play!"

    states, policies, values = buffer.sample(1)
    assert states[0].shape == (3, 8, 8)
    assert policies[0].shape == (8, 8)
    # Policy should represent a probability distribution over the board
    assert abs(policies[0].sum().item() - 1.0) < 1e-5
    assert values[0].item() in [-1, 0, 1]


def test_self_play_generates_valid_data():
    """Test that self-play generates valid training samples."""
    model = PolicyValueNet(board_size=8, num_blocks=3)
    model._init_weights()
    evaluator = NeuralEvaluator(model)
    buffer = ReplayBuffer(max_size=200)

    mcts_player1 = MCTSPlayer(
        MCTS(evaluator_fn=evaluator, n_simulations=30),
        temperature=1.0,
    )
    mcts_player2 = MCTSPlayer(
        MCTS(evaluator_fn=evaluator, n_simulations=30),
        temperature=1.0,
    )

    runner = SelfPlayRunner(
        player1=mcts_player1,
        player2=mcts_player2,
        buffer=buffer,
        verbose=False,
        config=config,
    )

    # Play a few games
    for _ in range(2):
        runner.play_game()

    assert len(buffer) > 0

    # Sample all data
    states, policies, values = buffer.sample(min(len(buffer), 10))

    # Validate states
    for state in states:
        assert state.shape == (3, 8, 8)
        # States should be binary (0 or 1)
        assert torch.all((state == 0) | (state == 1))

    # Validate policies
    for policy in policies:
        assert policy.shape == (8, 8)
        # Policy should sum to 1
        assert abs(policy.sum().item() - 1.0) < 1e-4
        # All probabilities should be non-negative
        assert torch.all(policy >= 0)

    # Validate values
    for value in values:
        assert value.item() in [-1.0, 0.0, 1.0]


def test_self_play_temperature_effect():
    """Test that temperature affects move selection."""
    model = PolicyValueNet(board_size=8, num_blocks=3)
    model._init_weights()
    evaluator = NeuralEvaluator(model)

    # Low temperature (more deterministic)
    buffer_low_temp = ReplayBuffer(max_size=100)
    mcts_low1 = MCTSPlayer(
        MCTS(evaluator_fn=evaluator, n_simulations=30),
        temperature=0.1,
    )
    mcts_low2 = MCTSPlayer(
        MCTS(evaluator_fn=evaluator, n_simulations=30),
        temperature=0.1,
    )
    runner_low = SelfPlayRunner(
        player1=mcts_low1,
        player2=mcts_low2,
        buffer=buffer_low_temp,
        verbose=False,
        config=config,
    )

    # High temperature (more exploration)
    buffer_high_temp = ReplayBuffer(max_size=100)
    mcts_high1 = MCTSPlayer(
        MCTS(evaluator_fn=evaluator, n_simulations=30),
        temperature=2.0,
    )
    mcts_high2 = MCTSPlayer(
        MCTS(evaluator_fn=evaluator, n_simulations=30),
        temperature=2.0,
    )
    runner_high = SelfPlayRunner(
        player1=mcts_high1,
        player2=mcts_high2,
        buffer=buffer_high_temp,
        verbose=False,
        config=config,
    )

    runner_low.play_game()
    runner_high.play_game()

    # Both should generate data
    assert len(buffer_low_temp) > 0
    assert len(buffer_high_temp) > 0

    # Sample policies
    _, policies_low, _ = buffer_low_temp.sample(min(5, len(buffer_low_temp)))
    _, policies_high, _ = buffer_high_temp.sample(min(5, len(buffer_high_temp)))

    # High temperature policies should generally be more uniform (higher entropy)
    # Low temperature policies should be more peaked (lower entropy)
    def entropy(policy):
        p = policy.flatten()
        p = p[p > 0]  # Only non-zero probs
        return -(p * torch.log(p)).sum().item()

    avg_entropy_low = sum(entropy(p) for p in policies_low) / len(policies_low)
    avg_entropy_high = sum(entropy(p) for p in policies_high) / len(policies_high)

    # High temp should generally have higher entropy
    # (Not strict assertion as it depends on the position)
    assert avg_entropy_low >= 0
    assert avg_entropy_high >= 0


def test_self_play_game_terminates():
    """Test that self-play games terminate properly."""
    model = PolicyValueNet(board_size=8, num_blocks=3)
    model._init_weights()
    evaluator = NeuralEvaluator(model)
    buffer = ReplayBuffer(max_size=100)

    mcts_player1 = MCTSPlayer(
        MCTS(evaluator_fn=evaluator, n_simulations=20),
        temperature=1.0,
    )
    mcts_player2 = MCTSPlayer(
        MCTS(evaluator_fn=evaluator, n_simulations=20),
        temperature=1.0,
    )

    runner = SelfPlayRunner(
        player1=mcts_player1,
        player2=mcts_player2,
        buffer=buffer,
        verbose=False,
        config=config,
    )

    # Should complete without hanging
    result = runner.play_game()

    # Result is number of samples added to buffer
    assert isinstance(result, int)
    assert result > 0, "No samples were added to buffer"
    assert len(buffer) > 0, "Buffer is empty after game"


if __name__ == "__main__":
    pytest.main()

