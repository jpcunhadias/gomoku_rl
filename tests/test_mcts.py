import pytest
import torch
from mcts.mcts import MCTS
from model.policy_value_net import PolicyValueNet
from mcts.neural_evaluator import NeuralEvaluator
from game.gomoku import GomokuBoard


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
    assert all(0.0 <= p <= 1.0 for p in action_probs.values()), "Probabilities must be in [0, 1]"
    assert pytest.approx(sum(action_probs.values()), abs=1e-3) == 1.0, "Probabilities do not sum to 1"

    # Check that the action probabilities correspond to legal moves
    legal_moves = board.get_legal_moves()
    legal_move_indices = [board.move_to_index(r, c) for r, c in legal_moves]
    assert set(action_probs.keys()).issubset(legal_move_indices), "Action keys should correspond to legal moves"

    print("MCTS test with Neural Evaluator passed!")
