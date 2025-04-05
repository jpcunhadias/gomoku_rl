import pytest

from mcts.mcts import MCTS


class DummyBoard:
    def __init__(self, state_id=0, terminal=False):
        self.state_id = state_id
        self.terminal = terminal
        self.moves = [1, 2, 3] if not terminal else []

    def clone(self):
        return DummyBoard(self.state_id, self.terminal)

    def apply_move(self, move):
        self.state_id += move  # simulate state change

    def is_terminal(self):
        return self.terminal

    def evaluate_terminal(self):
        return 1.0  # pretend current player wins


def dummy_evaluator(board):
    priors = [(1, 0.33), (2, 0.33), (3, 0.34)]
    value = 0.5
    return priors, value
å

def test_mcts_runs_and_returns_probs():
    mcts = MCTS(evaluator_fn=dummy_evaluator, c_puct=1.0, n_simulations=50)
    board = DummyBoard()

    action_probs = mcts.get_action_probs(board, temp=1.0)

    # Check the result is a valid probability distribution
    assert isinstance(action_probs, dict)
    assert all(0.0 <= p <= 1.0 for p in action_probs.values())
    assert pytest.approx(sum(action_probs.values()), abs=1e-3) == 1.0
    assert set(action_probs.keys()).issubset({1, 2, 3})
