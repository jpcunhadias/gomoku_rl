from typing import Optional, Dict, Tuple, List, Any

import numpy as np


class TreeNode:
    def __init__(
            self,
            parent: Optional["TreeNode"] = None,
            prior: float = 1.0,
            action_taken: Any = None,
    ):
        self.parent: Optional["TreeNode"] = parent
        self.children: Dict[Any, "TreeNode"] = {}
        self.n_visits: int = 0
        self.W: float = 0.0
        self.Q: float = 0.0
        self.P: float = prior
        self.action_taken: Any = action_taken

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_root(self) -> bool:
        return self.parent is None

    def expand(self, action_priors: List[Tuple[Any, float]], legal_moves: List[Any]):
        """Expand the tree node with legal action priors only."""
        legal_moves_set = set(legal_moves)

        for action, prob in action_priors:
            if action in legal_moves_set and action not in self.children:
                self.children[action] = TreeNode(
                    parent=self,
                    prior=prob,
                    action_taken=action,
                )

    def select_child(self, c_puct: float) -> Tuple[Any, "TreeNode"]:
        """Select child with highest PUCT score."""

        def puct_score(child: "TreeNode") -> float:
            u = c_puct * child.P * np.sqrt(self.n_visits + 1e-8) / (1 + child.n_visits)
            return child.Q + u

        return max(self.children.items(), key=lambda item: puct_score(item[1]))

    def update(self, value: float):
        """Update statistics with value from leaf evaluation."""
        self.n_visits += 1
        self.W += value
        self.Q = self.W / self.n_visits

    def backpropagate(self, value: float):
        """Recursively update current and ancestor nodes. Alternate sign for players."""
        if self.parent:
            self.parent.backpropagate(-value)
        self.update(value)
