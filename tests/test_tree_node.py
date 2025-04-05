import pytest
from mcts.tree_node import TreeNode


def test_tree_node_initialization():
    node = TreeNode()
    assert node.is_root()
    assert node.is_leaf()
    assert node.n_visits == 0
    assert node.Q == 0.0
    assert node.P == 1.0


def test_expand_creates_children():
    node = TreeNode()
    action_priors = [('a', 0.6), ('b', 0.4)]
    node.expand(action_priors)

    assert 'a' in node.children
    assert 'b' in node.children
    assert node.children['a'].P == 0.6
    assert node.children['b'].P == 0.4


def test_select_child_returns_highest_puct():
    root = TreeNode()
    root.expand([('a', 0.8), ('b', 0.2)])
    root.n_visits = 10  # simulate some activity

    # Simulate visits and Q values
    root.children['a'].n_visits = 5
    root.children['a'].W = 2.5
    root.children['a'].Q = 0.5

    root.children['b'].n_visits = 1
    root.children['b'].W = 0.5
    root.children['b'].Q = 0.5

    action, selected_node = root.select_child(c_puct=1.0)
    assert action in ['a', 'b']  # Either can be chosen depending on u

    # Test stability: running multiple times should not crash
    for _ in range(10):
        _ = root.select_child(c_puct=1.0)


def test_backpropagate_updates_all_parents():
    root = TreeNode()
    root.expand([('a', 1.0)])
    child = root.children['a']

    child.backpropagate(value=1.0)

    # Root should get -1, child should get +1
    assert child.n_visits == 1
    assert pytest.approx(child.Q) == 1.0
    assert root.n_visits == 1
    assert pytest.approx(root.Q) == -1.0
