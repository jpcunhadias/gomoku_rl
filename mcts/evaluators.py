import random
from typing import List, Tuple, Any
from typing import Union

import torch
from numpy import ndarray, dtype

from game.encoder import board_to_tensor
from game.gomoku import GomokuBoard
from model.policy_value_net import PolicyValueNet


class NeuralEvaluator:
    """Callable wrapper that evaluates boards using a policy-value network."""

    def __init__(self, model: PolicyValueNet, device: str = "cpu") -> None:
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def evaluate(self, board: Any) -> Tuple[List[Tuple[Any, float]], float]:
        """Return action priors and value for ``board``."""
        tensor_input = (
            board_to_tensor(board, board.current_player).unsqueeze(0).to(self.device)
        )

        with torch.no_grad():
            policy_logits, value = self.model(tensor_input)
            policy = torch.softmax(policy_logits.view(-1), dim=0).cpu().numpy()
            value = value.view(-1).item()

        legal_moves = board.get_legal_move_indices()
        action_priors = [(i, policy[i]) for i in legal_moves]
        return action_priors, value

    def __call__(
        self, board: Any
    ) -> tuple[
        list[tuple[Any, ndarray[Any, Union[dtype[Any], Any]]]], Union[int, float, bool]
    ]:
        """Evaluate ``board`` when used as a callable."""
        tensor_input = (
            board_to_tensor(board, board.current_player).unsqueeze(0).to(self.device)
        )

        board_size = board.board_size

        with torch.no_grad():
            policy_logits, value = self.model(tensor_input)

        policy = (
            torch.softmax(policy_logits.view(-1), dim=0)
            .view(board_size, board_size)
            .cpu()
            .numpy()
        )

        value = value.view(-1).item()

        legal_moves = board.get_legal_moves()
        action_priors = [(move, policy[move[0], move[1]]) for move in legal_moves]

        return action_priors, value


class ThreatRolloutEvaluator:
    """
    Evaluator for Pure MCTS that combines threat-based heuristics with ε-greedy rollouts.

    - Early game: uses threat heuristic directly.
    - Mid-to-late game: uses ε-greedy rollouts.
    - Always returns uniform priors and scalar value.
    """

    def __init__(
        self, rollout_depth: int = 20, num_rollouts: int = 2, epsilon: float = 0.1
    ):
        self.rollout_depth = rollout_depth
        self.num_rollouts = num_rollouts
        self.epsilon = epsilon

    def __call__(self, board: GomokuBoard) -> Tuple[List[Tuple[Any, float]], float]:
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            return [], 0.0

        # Uniform priors over legal moves
        prior = [(move, 1.0 / len(legal_moves)) for move in legal_moves]

        if board.is_terminal():
            return prior, self._terminal_value(board)

        move_count = board.get_num_moves()
        h = self._threat_score(board)
        r = self._rollout_value(board)
        alpha = 0.5 if move_count < 10 else 0.3  # more weight on heuristic early

        return prior, alpha * h + (1 - alpha) * r

    def _terminal_value(self, board: GomokuBoard) -> float:
        winner = board.get_winner()
        return 1.0 if winner == 1 else -1.0 if winner == 2 else 0.0

    def _threat_score(self, board: GomokuBoard) -> float:
        """
        Heuristic based on detecting open three/four threats.
        Score is normalized between [-1, 1].
        """

        def count_patterns(player: int) -> float:
            patterns = {
                "XXXX_": 1.0,  # open four
                "_XXXX": 1.0,
                "_XXX_": 0.6,  # open three
                "XXX__": 0.4,
                "__XXX": 0.4,
                "_XX_": 0.2,  # weak threat
            }
            score = 0.0

            # Translate board cells to 'X', 'O', or '.'
            symbol_map = {0: ".", player: "X", 3 - player: "O"}  # 3 - player swaps 1↔2
            for line in board.iter_lines():
                s = "".join(symbol_map[cell] for cell in line)
                for pat, w in patterns.items():
                    score += w * s.count(pat)
            return score

        player = board.current_player
        own_score = count_patterns(player)
        opp_score = count_patterns(3 - player)
        diff = own_score - opp_score
        return max(min(diff / 10.0, 1.0), -1.0)

    def _rollout_policy(self, board: GomokuBoard) -> Tuple[int, int]:
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            return None
        if random.random() > self.epsilon:
            # Choose move maximizing threat score
            best_move, best_val = None, -float("inf")
            for move in legal_moves:
                sim = board.clone()
                sim.apply_move(*move)
                val = self._threat_score(sim)
                if val > best_val:
                    best_move, best_val = move, val
            return best_move
        return random.choice(legal_moves)

    def _rollout_value(self, board: GomokuBoard) -> float:
        """Estimate value via ε-greedy rollout playouts."""
        results = []
        for _ in range(self.num_rollouts):
            sim = board.clone()
            for _ in range(self.rollout_depth):
                if sim.is_terminal():
                    break
                move = self._rollout_policy(sim)
                if move is None:
                    break
                sim.apply_move(*move)
            results.append(self._terminal_value(sim) if sim.is_terminal() else 0.0)
        return sum(results) / len(results)
