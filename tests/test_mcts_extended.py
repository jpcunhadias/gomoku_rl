import numpy as np

from game.gomoku import GomokuBoard
from mcts.evaluators import NeuralEvaluator
from mcts.mcts import MCTS
from model.policy_value_net import PolicyValueNet


def test_c_puct_schedule():
    schedule = {"enabled": True, "c0": 2.0, "lambda_": 0.5, "c_min": 0.5}
    mcts = MCTS(evaluator_fn=None, c_puct=1.0, c_puct_schedule=schedule)

    # At depth 0, c_puct should be base + c0
    assert mcts._effective_c_puct(0) == 1.0 + 2.0

    # At depth 1, it should be base + c0 * exp(-lambda * 1)
    assert mcts._effective_c_puct(1) == 1.0 + 2.0 * np.exp(-0.5 * 1)

    # At great depth, it should approach c_min
    assert mcts._effective_c_puct(100) > 0.5
    assert abs(mcts._effective_c_puct(100) - 1.0) < 1e-5  # approaches base c_puct, not c_min

    # Test c_min flooring
    schedule_with_c_min = {"enabled": True, "c0": -1.0, "lambda_": 0.5, "c_min": 0.8}
    mcts_with_c_min = MCTS(evaluator_fn=None, c_puct=0.1, c_puct_schedule=schedule_with_c_min)
    assert mcts_with_c_min._effective_c_puct(0) == 0.8


def test_dirichlet_noise():
    board = GomokuBoard(board_size=3)
    board.apply_move(1, 1)

    model = PolicyValueNet(board_size=3, num_blocks=1)
    model._init_weights()
    evaluator = NeuralEvaluator(model)
    mcts = MCTS(evaluator_fn=evaluator, c_puct=1.0, n_simulations=1)

    # Run one simulation to expand the root
    mcts.get_action_probs(board, temp=1.0)

    # Get the original priors
    original_priors = {a: c.P for a, c in mcts.root.children.items()}

    # Apply Dirichlet noise
    mcts.apply_root_dirichlet(epsilon=0.25, alpha=0.5)

    # Get the new priors
    new_priors = {a: c.P for a, c in mcts.root.children.items()}

    # Check that the priors have been modified
    assert original_priors != new_priors

    # Check that the sum of new priors is still close to 1
    assert np.isclose(sum(new_priors.values()), 1.0)

    # Check that each new prior is a mix of the old prior and some noise
    for action, old_prior in original_priors.items():
        new_prior = new_priors[action]
        assert new_prior >= (1 - 0.25) * old_prior


def test_root_noise_in_get_action_probs():
    board = GomokuBoard(board_size=3)
    board.apply_move(1, 1)

    model = PolicyValueNet(board_size=3, num_blocks=1)
    model._init_weights()
    evaluator = NeuralEvaluator(model)
    mcts = MCTS(evaluator_fn=evaluator, c_puct=1.0, n_simulations=1)

    # Get action probabilities without noise
    mcts.get_action_probs(board, temp=1.0)
    priors_no_noise = {a: c.P for a, c in mcts.root.children.items()}

    # Reset root and get action probabilities with noise
    mcts.reset_root()
    mcts.get_action_probs(board, temp=1.0, root_noise=(0.25, 0.5))
    priors_with_noise = {a: c.P for a, c in mcts.root.children.items()}

    # Check that the priors have been modified
    assert priors_no_noise != priors_with_noise


def test_backpropagation():
    board = GomokuBoard(board_size=3)
    board.apply_move(1, 1)

    # Mock evaluator that returns a fixed value and priors
    def mock_evaluator(board):
        legal_moves = board.get_legal_moves()
        priors = {move: 1.0 / (i + 2) for i, move in enumerate(legal_moves)}
        total = sum(priors.values())
        priors = {move: p / total for move, p in priors.items()}
        return list(priors.items()), 0.5

    mcts = MCTS(evaluator_fn=mock_evaluator, c_puct=1.0, n_simulations=2)

    # Run one simulation to expand the root
    mcts.run_simulation(mcts.root, board.clone())

    # The root should have been visited once, and its Q value should be 0.5
    assert mcts.root.n_visits == 1
    assert mcts.root.Q == 0.5

    # The children are created, but not visited yet
    for child in mcts.root.children.values():
        assert child.n_visits == 0
        assert child.Q == 0

    # Run a second simulation. This will select a child and backpropagate the value.
    mcts.run_simulation(mcts.root, board.clone())

    # The root should have been visited twice
    assert mcts.root.n_visits == 2

    # One of the children should have been visited
    visited_children = [c for c in mcts.root.children.values() if c.n_visits > 0]
    assert len(visited_children) == 1
    selected_child = visited_children[0]

    # The selected child should have been visited once, and its Q value should be 0.5
    assert selected_child.n_visits == 1
    assert selected_child.Q == 0.5

    # The root's Q value should be the average of the two simulations, taking into account the alternating player
    assert mcts.root.Q == (0.5 - 0.5) / 2


def test_tree_reuse():
    board = GomokuBoard(board_size=3)
    board.apply_move(1, 1)

    model = PolicyValueNet(board_size=3, num_blocks=1)
    model._init_weights()
    evaluator = NeuralEvaluator(model)
    mcts = MCTS(evaluator_fn=evaluator, c_puct=1.0, n_simulations=10)

    # Run some simulations
    mcts.get_action_probs(board, temp=1.0)

    # Get the root and its children
    old_root = mcts.root
    children = old_root.children

    # Choose a move and update the MCTS
    action, _ = max(children.items(), key=lambda item: item[1].n_visits)
    mcts.update_with_move(action)

    # The new root should be the child of the old root
    assert mcts.root == children[action]
    assert mcts.root.parent is None


def _normalized_entropy(probs: dict) -> float:
    values = np.array(list(probs.values()), dtype=np.float64)
    values = values[values > 1e-12]
    if len(values) <= 1:
        return 0.0
    h = float(-(values * np.log(values)).sum())
    return h / np.log(len(values))


def test_normalize_counts_temperature_effect():
    """Deterministic version of what tests/test_self_play.py's temperature-effect test was
    trying (and failing) to assert with real self-play noise: this tests _normalize_counts
    directly on fixed counts, so the claim is checked exactly rather than hoping randomness
    cooperates."""
    mcts = MCTS(evaluator_fn=None, c_puct=1.0)
    counts = {0: 100, 1: 10, 2: 1}

    low_temp_probs = mcts._normalize_counts(counts, temp=0.3)
    high_temp_probs = mcts._normalize_counts(counts, temp=2.0)

    for probs in (low_temp_probs, high_temp_probs):
        assert abs(sum(probs.values()) - 1.0) < 1e-6
        assert all(p >= 0 for p in probs.values())

    assert _normalized_entropy(low_temp_probs) < _normalized_entropy(high_temp_probs)
    # Sharper (lower temp) should also concentrate more mass on the most-visited action.
    assert low_temp_probs[0] > high_temp_probs[0]


def test_normalize_counts_near_zero_temp_is_deterministic_argmax():
    mcts = MCTS(evaluator_fn=None, c_puct=1.0)
    counts = {0: 5, 1: 20, 2: 3}

    probs = mcts._normalize_counts(counts, temp=1e-4)

    assert probs[1] == 1.0
    assert probs[0] == 0.0
    assert probs[2] == 0.0


def test_normalize_counts_high_exponent_log_space_path_is_valid():
    """temp small enough that 1/temp > 10 takes the log-space branch (overflow avoidance
    for large counts) - never exercised by any other test."""
    mcts = MCTS(evaluator_fn=None, c_puct=1.0)
    counts = {0: 500, 1: 50, 2: 5}

    probs = mcts._normalize_counts(counts, temp=0.05)  # exponent = 20

    assert abs(sum(probs.values()) - 1.0) < 1e-6
    assert not any(np.isnan(p) for p in probs.values())
    # Still ordered by visit count, just very sharply.
    assert probs[0] > probs[1] > probs[2]


def test_normalize_counts_all_zero_visits_falls_back_to_uniform():
    """Regression test for a real bug: dividing by sum(counts)==0 used to silently produce
    NaN probabilities (visible as a RuntimeWarning in every test run before this was fixed)."""
    mcts = MCTS(evaluator_fn=None, c_puct=1.0)
    counts = {0: 0, 1: 0, 2: 0}

    probs = mcts._normalize_counts(counts, temp=1.0)

    assert not any(np.isnan(p) for p in probs.values())
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    assert probs[0] == probs[1] == probs[2]
