import torch
from typing import Any, List, Tuple

from game.encoder import board_to_tensor
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

        legal_moves = board.get_legal_move_indices()
        action_priors = [(i, policy[i]) for i in legal_moves]
        return action_priors, value.item()

    def __call__(self, board: Any) -> Tuple[List[Tuple[Any, float]], float]:
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

        legal_moves = board.get_legal_moves()
        action_priors = [(move, policy[move[0], move[1]]) for move in legal_moves]

        return action_priors, value.item()
