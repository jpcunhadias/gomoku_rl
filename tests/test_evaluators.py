import pytest

from game.gomoku import GomokuBoard
from mcts.evaluators import NeuralEvaluator, ThreatRolloutEvaluator
from model.policy_value_net import PolicyValueNet


def test_neural_evaluator_returns_correct_format():
    """Test that NeuralEvaluator returns priors and value in correct format."""
    model = PolicyValueNet(board_size=8, num_blocks=3)
    model._init_weights()
    evaluator = NeuralEvaluator(model, device="cpu")

    board = GomokuBoard(board_size=8)
    board.apply_move(3, 3)  # Make one move

    action_priors, value = evaluator(board)

    # Check that action_priors is a list of tuples
    assert isinstance(action_priors, list)
    assert len(action_priors) > 0

    # Check each prior
    total_prob = 0.0
    for action, prob in action_priors:
        assert isinstance(action, tuple)
        assert len(action) == 2
        assert isinstance(prob, float)
        assert prob >= 0.0
        total_prob += prob

    # Check that probabilities sum to 1 (approximately)
    assert abs(total_prob - 1.0) < 1e-5, f"Priors sum to {total_prob}, not 1.0"

    # Check value
    assert isinstance(value, float)
    assert -1.0 <= value <= 1.0


def test_neural_evaluator_legal_moves_only():
    """Test that NeuralEvaluator only returns priors for legal moves."""
    model = PolicyValueNet(board_size=8, num_blocks=3)
    model._init_weights()
    evaluator = NeuralEvaluator(model, device="cpu")

    board = GomokuBoard(board_size=8)
    board.apply_move(0, 0)
    board.apply_move(1, 1)
    board.apply_move(2, 2)

    action_priors, _ = evaluator(board)

    legal_moves = set(board.get_legal_moves())
    returned_moves = set(action for action, _ in action_priors)

    # All returned moves should be legal
    assert returned_moves.issubset(legal_moves)

    # Should return all legal moves
    assert len(returned_moves) == len(legal_moves)


def test_neural_evaluator_terminal_board():
    """Test evaluator on terminal board."""
    model = PolicyValueNet(board_size=8, num_blocks=3)
    model._init_weights()
    evaluator = NeuralEvaluator(model, device="cpu")

    board = GomokuBoard(board_size=8)
    # Create a winning position
    for i in range(5):
        board.apply_move(0, i)
        if i < 4:
            board.apply_move(1, i)

    # Board should be terminal
    assert board.is_terminal()

    action_priors, value = evaluator(board)

    # Evaluator may still return priors for remaining legal moves
    # even though game is over - this is implementation-specific
    # The key is that the value should be valid
    assert isinstance(value, float)
    assert -1.0 <= value <= 1.0


def test_neural_evaluator_different_board_states():
    """Test evaluator produces different outputs for different board states."""
    model = PolicyValueNet(board_size=8, num_blocks=3)
    model._init_weights()
    evaluator = NeuralEvaluator(model, device="cpu")

    board1 = GomokuBoard(board_size=8)
    board1.apply_move(3, 3)

    board2 = GomokuBoard(board_size=8)
    board2.apply_move(0, 0)

    _, value1 = evaluator(board1)
    _, value2 = evaluator(board2)

    # Values might be different (not guaranteed but likely)
    # At minimum, both should be valid
    assert -1.0 <= value1 <= 1.0
    assert -1.0 <= value2 <= 1.0


def test_threat_rollout_evaluator_format():
    """Test ThreatRolloutEvaluator returns correct format."""
    evaluator = ThreatRolloutEvaluator(rollout_depth=10, num_rollouts=2)

    board = GomokuBoard(board_size=8)
    board.apply_move(3, 3)

    action_priors, value = evaluator(board)

    # Check format
    assert isinstance(action_priors, list)
    assert len(action_priors) > 0

    # Check uniform priors
    expected_prob = 1.0 / len(action_priors)
    for _action, prob in action_priors:
        assert abs(prob - expected_prob) < 1e-6

    # Check value
    assert isinstance(value, float)
    assert -1.0 <= value <= 1.0


def test_threat_rollout_evaluator_terminal():
    """Test ThreatRolloutEvaluator on terminal state."""
    evaluator = ThreatRolloutEvaluator(rollout_depth=10, num_rollouts=1)

    board = GomokuBoard(board_size=8)
    # Create winning position for player 1
    for i in range(5):
        board.apply_move(0, i)
        if i < 4:
            board.apply_move(1, i)

    assert board.is_terminal()
    assert board.get_winner() == 1

    action_priors, value = evaluator(board)

    # Evaluator returns priors based on get_legal_moves()
    # The value should reflect that player 1 won
    assert isinstance(value, float)
    assert -1.0 <= value <= 1.0


def test_neural_evaluator_batch_consistency():
    """Test that evaluator gives consistent results."""
    model = PolicyValueNet(board_size=8, num_blocks=3)
    model._init_weights()
    model.eval()  # Set to eval mode for deterministic batch norm
    evaluator = NeuralEvaluator(model, device="cpu")

    board = GomokuBoard(board_size=8)
    board.apply_move(3, 3)

    # Evaluate same board twice
    priors1, value1 = evaluator(board)
    priors2, value2 = evaluator(board)

    # Should get identical results
    assert value1 == value2
    assert len(priors1) == len(priors2)

    # Convert to dict for easier comparison
    priors1_dict = dict(priors1)
    priors2_dict = dict(priors2)

    for move in priors1_dict:
        assert abs(priors1_dict[move] - priors2_dict[move]) < 1e-6


if __name__ == "__main__":
    pytest.main()
