import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Identity connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Skip connection
        return F.relu(out)


class ValueClassifierHead(nn.Module):
    def __init__(self, input_size=15 * 15, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PolicyValueNet(nn.Module):
    def __init__(self, board_size=15, num_blocks=5):
        super(PolicyValueNet, self).__init__()

        # Initial Conv Layer
        self.conv_input = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(64)

        # Residual Blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64, 64) for _ in range(num_blocks)]
        )

        # Policy Head
        self.policy_conv = nn.Conv2d(64, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)

        # Value Head
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.value_head = ValueClassifierHead(input_size=board_size * board_size)

        # Initialize weights
        self._init_weights()

    def forward(self, x):
        # Input Layer
        x = F.relu(self.bn_input(self.conv_input(x)))

        # Residual Blocks
        x = self.residual_blocks(x)

        # Policy Head
        policy = self.policy_conv(x)
        policy = policy.view(policy.size(0), -1)  # Flatten the output
        policy = self.policy_fc(policy)

        # Value Head
        value = self.value_conv(x)
        value = self.value_head(value)

        return policy, value

    def extract_value_features(self, x):
        x = F.relu(self.bn_input(self.conv_input(x)))
        x = self.residual_blocks(x)
        value = self.value_conv(x)
        return value.view(value.size(0), -1)  # Flatten

    @classmethod
    def load_from_checkpoint(cls, path, board_size=15, num_blocks=5, device=None):
        """
        Loads a model from a checkpoint that contains 'model_state_dict'.
        Args:
            path (str): Path to the saved .pth file
            board_size (int): Board size used during training
            num_blocks (int): Number of residual blocks
            device (str or torch.device): 'cpu' or 'cuda'
        Returns:
            PolicyValueNet instance with loaded weights
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location=device)
        model = cls(board_size=board_size, num_blocks=num_blocks)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()  # Optional: set to eval mode by default
        return model

    def _init_weights(self):
        nn.init.xavier_uniform_(self.policy_fc.weight)
        nn.init.zeros_(self.policy_fc.bias)

        # Access the submodule inside value_head
        nn.init.xavier_uniform_(self.value_head.fc2.weight)
        nn.init.zeros_(self.value_head.fc2.bias)


if __name__ == "__main__":
    ...
