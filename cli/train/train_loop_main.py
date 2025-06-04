import torch

from train.replay_buffer import ReplayBuffer
from model.policy_value_net import PolicyValueNet
from train.config import get_config
from train.train_loop import AlphaZeroTrainer

# Load configuration
config = get_config()


def main() -> None:
    """Entry point for running the full AlphaZero training loop."""
    print("Starting AlphaZero training loop")

    # Select device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize model
    model = PolicyValueNet()  # update if your net requires more args

    replay_buffer = ReplayBuffer.load("checkpoints/checkpoints_old/replay_buffer.pkl")
    print(f"Loaded buffer with {len(replay_buffer)} samples")

    # Initialize trainer
    trainer = AlphaZeroTrainer(model, replay_buffer, config, device=device)

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
