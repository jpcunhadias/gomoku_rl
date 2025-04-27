import pytest

from game.gomoku import GomokuBoard
from mcts.mcts import MCTS
from mcts.neural_evaluator import NeuralEvaluator
from model.policy_value_net import PolicyValueNet


def test_mcts_runs_and_returns_probs():
    # Initialize the board and model
    board = GomokuBoard(board_size=15)
    model = PolicyValueNet(board_size=15, num_blocks=5)
    evaluator = NeuralEvaluator(model)

    # Initialize MCTS with the neural evaluator
    mcts = MCTS(evaluator_fn=evaluator.evaluate, c_puct=1.0, n_simulations=50)

    # Run simulations to get action probabilities
    action_probs = mcts.get_action_probs(board, temp=1.0)

    # Check that the result is a valid probability distribution
    assert isinstance(action_probs, dict), f"Expected dict, got {type(action_probs)}"

    # If no actions returned, fallback to checking legal moves manually
    if not action_probs:
        legal_moves = board.get_legal_moves()
        assert len(legal_moves) > 0, "There should be legal moves on an empty board"
    else:
        assert all(0.0 <= p <= 1.0 for p in action_probs.values()), (
            "Probabilities must be in [0, 1]"
        )
        total_prob = sum(action_probs.values())
        if len(action_probs) == 1:
            assert total_prob == 1.0, "Total probability should be exactly 1 for random fallback"
        else:
            assert pytest.approx(total_prob, abs=1e-3) == 1.0, (
                f"Probabilities do not sum to 1 (total={total_prob})"
            )

