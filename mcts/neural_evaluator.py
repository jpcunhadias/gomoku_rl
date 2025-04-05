import torch

from game.encoder import board_to_tensor
from model.policy_value_net import PolicyValueNet


class NeuralEvaluator:
    def __init__(self, model: PolicyValueNet):
        self.model = model

    def evaluate(self, board) -> (list, float):
        """
        Evaluates a given board using the neural network model.

        Args:
            board: A GomokuBoard instance representing the current state of the game.

        Returns:
            action_priors (list): A list of tuples (action, probability) for each move.
            value (float): A scalar value representing the board's evaluation.
        """
        # Convert board to tensor input for the model
        tensor_input = board_to_tensor(board, board.current_player)

        # Ensure we have a 4D tensor [batch_size, channels, height, width]
        if tensor_input.dim() == 3:
            tensor_input = tensor_input.unsqueeze(0)

        # Set the model to evaluation mode
        self.model.eval()

        with torch.no_grad():
            # Forward pass through the model
            policy_logits, value = self.model(tensor_input)

            # Apply softmax to convert logits to probabilities
            policy = torch.softmax(policy_logits, dim=-1).cpu().numpy()  # shape: [1, num_moves]

            # Remove batch dimension
            policy = policy[0]  # shape: [num_moves]

            # Convert to list of (action, probability) pairs
            legal_moves = board.get_legal_move_indices()
            action_priors = [(i, policy[i]) for i in legal_moves]

            # Return the action priors and scalar value
            return action_priors, value.item()
