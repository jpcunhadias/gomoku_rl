import random
from typing import Callable, Dict, Any

import numpy as np

from mcts.tree_node import TreeNode


class MCTS:
    """Monte-Carlo Tree Search with optional RAVE support."""

    def __init__(
        self,
        evaluator_fn: Callable[[Any], Any],
        c_puct: float = 1.0,
        n_simulations: int = 800,
        use_rave: bool = True,
    ) -> None:
        """
        Args:
            evaluator_fn: Function that evaluates a board and returns (priors, value).
            c_puct: Exploration constant.
            n_simulations: Simulations per move.
            use_rave: Whether to use RAVE enhancement.
        """
        self.evaluator_fn = evaluator_fn
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.use_rave = use_rave
        self.root = TreeNode(use_rave=use_rave)
        self.last_root_visit_counts: Dict[Any, int] | None = None
        self._root_noise_applied = False

    def update_with_move(self, move: Any) -> None:
        """Reuse the subtree rooted at ``move`` if it exists."""
        if move in self.root.children:
            self.root = self.root.children[move]
            self.root.parent = None
        else:
            self.root = TreeNode(use_rave=self.use_rave)

    def set_simulation_budget(self, n: int) -> None:
        self.n_simulations = int(n)

    def run_simulation(self, root: TreeNode, board: Any) -> None:
        node = root
        state = board.clone()
        visited_moves = set()

        if node == self.root:
            legal_moves_set = set(state.get_legal_moves())
            illegal_children = [
                action for action in node.children if action not in legal_moves_set
            ]
            for action in illegal_children:
                del node.children[action]

        while not node.is_leaf():
            legal_moves_set = set(state.get_legal_moves())
            for action in list(node.children):
                if action not in legal_moves_set:
                    del node.children[action]

            if node.is_leaf():
                break

            action, node = node.select_child(
                c_puct=self.c_puct,
                k_rave=300.0,
            )

            if isinstance(action, int):
                row, col = state.index_to_move(action)
            elif isinstance(action, tuple):
                row, col = action
            else:
                raise TypeError("Action must be int or (row, col) tuple.")

            if not state.is_legal_move(row, col):
                print(f"[DEBUG] Illegal move selected: {action} → ({row}, {col})")
                state.render()
                raise RuntimeError("MCTS selected an illegal move.")

            state.apply_move(row, col)
            visited_moves.add((row, col))

        if state.is_terminal():
            value = state.evaluate_terminal()
        else:
            priors, value = self.evaluator_fn(state)
            legal_moves = state.get_legal_moves()
            node.expand(
                priors,
                legal_moves,
            )

        node.backpropagate(value, visited_moves)

    def apply_root_dirichlet(self, epsilon: float, alpha: float) -> None:
        """Mix Dir(α) into current root's priors: P <- (1-ε)P + ε·Dir(α)."""
        node = self.root
        if not node.children:
            return
        actions = list(node.children.keys())
        noise = np.random.dirichlet([alpha] * len(actions))
        for a, n in zip(actions, noise):
            node.children[a].P = (1.0 - epsilon) * node.children[a].P + epsilon * n

    def get_action_probs(
        self, board, temp: float = 1e-3, root_noise: tuple[float, float] | None = None
    ):
        # Ensure root is expanded at least once
        if not self.root.children:
            self.run_simulation(self.root, board)

        # Apply root noise once (on priors) if requested
        if root_noise is not None:
            # (optional) guard — helps catch accidental double application
            if self._root_noise_applied:
                # You can downgrade to debug/trace if you prefer
                # print("[WARN] root noise requested more than once; ignoring.")
                pass
            else:
                eps, alpha = root_noise
                if eps > 0.0 and alpha > 0.0:
                    self.apply_root_dirichlet(eps, alpha)
                self._root_noise_applied = True

        # Do remaining simulations
        for _ in range(self.n_simulations - 1):
            self.run_simulation(self.root, board)

        visit_counts = {a: ch.n_visits for a, ch in self.root.children.items()}
        self.last_root_visit_counts = visit_counts  # <-- so root_visit_stats() works
        return self._normalize_counts(visit_counts, temp)

    def reset_root(self) -> None:
        """Reset the root node to an empty state."""
        self.root = TreeNode(use_rave=self.use_rave)
        self.last_root_visit_counts = None
        self._root_noise_applied = False

    def _normalize_counts(
        self, counts: Dict[Any, int], temp: float
    ) -> Dict[Any, float]:
        if not counts:
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

    def root_visit_stats(self):
        vc = (
            list(self.last_root_visit_counts.values())
            if self.last_root_visit_counts
            else []
        )
        if not vc:
            return None
        return {
            "n_children": len(vc),
            "min": int(min(vc)),
            "max": int(max(vc)),
            "mean": float(sum(vc) / len(vc)),
        }
