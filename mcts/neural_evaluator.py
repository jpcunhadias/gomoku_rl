import torch

from game.encoder import board_to_tensor
from model.policy_value_net import PolicyValueNet


class NeuralEvaluator:
    def __init__(self, model: PolicyValueNet, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def evaluate(self, board) -> (list, float):
        """
        Explicitly evaluates a GomokuBoard, returning legal move priors and value.
        """
        tensor_input = board_to_tensor(board, board.current_player).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_logits, value = self.model(tensor_input)
            policy = torch.softmax(policy_logits.view(-1), dim=0).cpu().numpy()

        legal_moves = board.get_legal_move_indices()
        action_priors = [(i, policy[i]) for i in legal_moves]
        return action_priors, value.item()

    def __call__(self, board):
        """
        Evaluates a board using the policy-value network.
        Returns a list of (legal_move, probability) pairs and a scalar value.
        """
        tensor_input = board_to_tensor(board, board.current_player).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_logits, value = self.model(tensor_input)

        policy = torch.softmax(policy_logits.view(-1), dim=0).view(15, 15).cpu().numpy()

        legal_moves = board.get_legal_moves()
        action_priors = [(move, policy[move[0], move[1]]) for move in legal_moves]

        return action_priors, value.item()
