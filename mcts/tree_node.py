from collections import defaultdict
from typing import Optional, Dict, Tuple, List, Set, Any

import numpy as np


class TreeNode:
    """Node used in the Monte-Carlo search tree with optional RAVE support."""

    def __init__(
        self,
        parent: Optional["TreeNode"] = None,
        prior: float = 1.0,
        action_taken: Any = None,
        use_rave: bool = True,
    ) -> None:
        self.parent: Optional["TreeNode"] = parent
        self.children: Dict[Any, "TreeNode"] = {}
        self.n_visits: int = 0
        self.W: float = 0.0
        self.Q: float = 0.0
        self.P: float = prior
        self.action_taken: Any = action_taken
        self.use_rave: bool = use_rave

        # RAVE statistics
        self.n_rave: Dict[Any, int] = defaultdict(int)
        self.w_rave: Dict[Any, float] = defaultdict(float)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_root(self) -> bool:
        return self.parent is None

    def expand(
        self, action_priors, legal_moves, debug=False, prior_exponent_beta: float = 1.0
    ):
        legal_moves_set = set(legal_moves)

        # Build a dict of priors for legal moves only
        p = {a: prob for a, prob in action_priors if a in legal_moves_set}
        if not p:
            return  # nothing to expand

        # β-sharpening over legal
        if prior_exponent_beta != 1.0:
            # raise and renormalize over LEGAL only
            arr = np.array(list(p.values()), dtype=np.float32)
            arr = np.power(arr, prior_exponent_beta)
            s = float(arr.sum())
            if s > 0:
                arr /= s
            for (k, _), v in zip(p.items(), arr):
                p[k] = float(v)

        # create children with (possibly) sharpened P
        for action, prob in p.items():
            if action not in self.children:
                self.children[action] = TreeNode(
                    parent=self, prior=prob, action_taken=action, use_rave=self.use_rave
                )
            elif debug and action not in legal_moves_set:
                print(f"[DEBUG] Ignored invalid expansion action: {action}")

    def select_child(
        self, c_puct: float, k_rave: float = 300.0
    ) -> Tuple[Any, "TreeNode"]:
        """Select child with highest (PUCT + RAVE) or standard PUCT score."""
        if self.is_leaf():
            raise ValueError("Cannot select child from a leaf node.")

        def puct_score(child: "TreeNode", move: Any) -> float:
            u = c_puct * child.P * np.sqrt(self.n_visits + 1e-8) / (1 + child.n_visits)
            Q = child.Q

            if self.use_rave:
                n_rave = self.n_rave[move]
                w_rave = self.w_rave[move]
                Q_rave = w_rave / n_rave if n_rave > 0 else 0.0
                beta = n_rave / (n_rave + child.n_visits + k_rave + 1e-8)
                Q_blend = beta * Q_rave + (1 - beta) * Q
                return Q_blend + u
            else:
                return Q + u

        return max(self.children.items(), key=lambda item: puct_score(item[1], item[0]))

    def update(self, value: float) -> None:
        self.n_visits += 1
        self.W += value
        self.Q = self.W / self.n_visits

    def update_rave(self, move: Any, value: float) -> None:
        self.n_rave[move] += 1
        self.w_rave[move] += value

    def backpropagate(
        self, value: float, visited_moves: Optional[Set[Any]] = None
    ) -> None:
        """
        Backpropagate value up the tree, updating parent nodes and RAVE statistics.
        """
        if self.parent:
            self.parent.backpropagate(-value, visited_moves)

        self.update(value)

        if visited_moves is not None and self.use_rave:
            for move in visited_moves:
                self.update_rave(move, value)
