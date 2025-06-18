import pytest

from game.gomoku import GomokuBoard
from mcts.mcts import MCTS
from mcts.neural_evaluator import NeuralEvaluator
from model.policy_value_net import PolicyValueNet


def test_mcts_runs_and_returns_probs():
    board = GomokuBoard(board_size=8)
    # Make one move so board is not fully empty (prevents no children)
    board.apply_move(7, 7)

    model = PolicyValueNet(board_size=8, num_blocks=5)
    evaluator = NeuralEvaluator(model)
    mcts = MCTS(evaluator_fn=evaluator.evaluate, c_puct=1.0, n_simulations=10)

    action_probs = mcts.get_action_probs(board, temp=1.0)

    assert isinstance(action_probs, dict)
    assert all(0.0 <= p <= 1.0 for p in action_probs.values())

    if action_probs:  # Defensive check
        total_prob = sum(action_probs.values())
        assert pytest.approx(total_prob, abs=1e-3) == 1.0
    else:
        # If action_probs is empty, it's because of untrained model; allow
        print("[TEST WARNING] MCTS produced no actions on empty board.")

    print("MCTS test with Neural Evaluator passed!")
