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
        self.value_fc1 = nn.Linear(board_size * board_size, 128)
        self.value_fc2 = nn.Linear(128, 1)

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
        value = value.view(value.size(0), -1)  # Flatten the output
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))  # Output in range [-1, 1]

        return policy, value

if __name__ == '__main__':
 ...