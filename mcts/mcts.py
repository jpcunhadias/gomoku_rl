import random
from typing import Callable, Dict, Any

import numpy as np

from mcts.tree_node import TreeNode


class MCTS:
    def __init__(
        self, evaluator_fn: Callable, c_puct: float = 1.0, n_simulations: int = 800
    ):
        """
        evaluator_fn: Callable that takes a board and returns (priors_dict, value)
        """
        self.evaluator_fn = evaluator_fn
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.root = TreeNode()

    def update_with_move(self, move):
        """
        Reuse the subtree rooted at the selected child node (if available).
        """
        if hasattr(self, "root") and move in self.root.children:
            self.root = self.root.children[move]
            self.root.parent = None  # clear reference to previous root
        else:
            self.root = TreeNode()  # reset tree if move not in children

    def run_simulation(self, root: TreeNode, board):
        node = root
        state = board.clone()

        # Clean up root's children if board has changed
        if node == self.root:
            legal_moves_set = set(state.get_legal_moves())
            illegal_children = [action for action in node.children if action not in legal_moves_set]
            for action in illegal_children:
                del node.children[action]

        # Selection
        while not node.is_leaf():
            action, node = node.select_child(self.c_puct)
            if isinstance(action, int):
                row, col = state.index_to_move(action)
            elif isinstance(action, tuple):
                row, col = action
            else:
                raise TypeError("Action must be either an int or a (row, col) tuple.")

            if not state.is_legal_move(row, col):
                print(f"[DEBUG] Illegal move selected: {action} → ({row}, {col})")
                print("[DEBUG] Current board:")
                state.render()
                print("[DEBUG] Legal moves:", state.get_legal_moves())
                raise RuntimeError("MCTS selected an illegal move during simulation.")

            state.apply_move(row, col)

        # Check for terminal state
        if state.is_terminal():
            value = state.evaluate_terminal()  # +1 win, -1 loss, 0 draw
        else:
            priors, value = self.evaluator_fn(state)

            legal_moves = state.get_legal_moves()

            node.expand(priors, legal_moves)

        node.backpropagate(value)

    def get_action_probs(self, board, temp: float = 1e-3) -> Dict[Any, float]:
        """
        Run simulations and return a distribution over actions from the root node.
        """

        for _ in range(self.n_simulations):
            self.run_simulation(self.root, board)

        visit_counts = {
            action: child.n_visits for action, child in self.root.children.items()
        }

        return self._normalize_counts(visit_counts, temp)

    def _normalize_counts(
        self, counts: Dict[Any, int], temp: float
    ) -> Dict[Any, float]:
        if not counts:
            # No legal moves available
            return {}

        if temp <= 1e-3:
            max_visits = max(counts.values())
            best_actions = [a for a, v in counts.items() if v == max_visits]
            best_action = random.choice(best_actions)
            return {a: 1.0 if a == best_action else 0.0 for a in counts}

        counts_arr = np.array(list(counts.values()), dtype=np.float32)
        actions = list(counts.keys())
        counts_arr = np.power(counts_arr, 1.0 / temp)
        probs = counts_arr / np.sum(counts_arr)
        return dict(zip(actions, probs))
