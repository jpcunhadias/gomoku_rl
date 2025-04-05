from typing import Callable, Dict, Any
from mcts.tree_node import TreeNode
import numpy as np
import random


class MCTS:
    def __init__(self, evaluator_fn: Callable, c_puct: float = 1.0, n_simulations: int = 800):
        """
        evaluator_fn: Callable that takes a board and returns (priors_dict, value)
        """
        self.evaluator_fn = evaluator_fn
        self.c_puct = c_puct
        self.n_simulations = n_simulations

    def run_simulation(self, root: TreeNode, board):
        """
        Runs a single MCTS simulation from the given root using the current board state.
        """
        node = root
        state = board.clone()

        # Selection
        while not node.is_leaf():
            action, node = node.select_child(self.c_puct)
            # If action is an integer, convert it to (row, col)
            if isinstance(action, int):
                row, col = state.index_to_move(action)
            # Otherwise, if it's already a tuple, use it directly
            elif isinstance(action, tuple):
                row, col = action
            else:
                raise TypeError("Action must be either an int or a tuple of (row, col)")

            state.apply_move(row, col)

        # Check for terminal state before expansion
        if state.is_terminal():
            value = state.evaluate_terminal()  # +1 win, -1 loss, 0 draw
        else:
            # Expansion & evaluation
            priors, value = self.evaluator_fn(state)
            node.expand(priors)

        # Backpropagate
        node.backpropagate(value)

    def get_action_probs(self, board, temp: float = 1e-3) -> Dict[Any, float]:
        """
        Run simulations and return a distribution over actions from the root node.
        """
        root = TreeNode()

        for _ in range(self.n_simulations):
            self.run_simulation(root, board)

        # Build {action: visit_count}
        visit_counts = {action: child.n_visits for action, child in root.children.items()}
        return self._normalize_counts(visit_counts, temp)

    def _normalize_counts(self, counts: Dict[Any, int], temp: float) -> Dict[Any, float]:
        if temp <= 1e-3:
            # Deterministic choice
            max_visits = max(counts.values())
            best_actions = [a for a, v in counts.items() if v == max_visits]
            best_action = random.choice(best_actions)
            return {a: 1.0 if a == best_action else 0.0 for a in counts}

        counts_arr = np.array(list(counts.values()), dtype=np.float32)
        actions = list(counts.keys())
        counts_arr = np.power(counts_arr, 1.0 / temp)
        probs = counts_arr / np.sum(counts_arr)
        return dict(zip(actions, probs))
