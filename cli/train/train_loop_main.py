import os

import torch

from model.policy_value_net import PolicyValueNet
from train.config import get_config
from train.replay_buffer import ReplayBuffer
from train.train_loop import AlphaZeroTrainer

# Load configuration
config = get_config()


def main() -> None:
    """Entry point for running the full AlphaZero training loop."""
    print("Starting AlphaZero training loop")

    # Select device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    checkpoint_path = "checkpoints/policy_value_net_best.pth"
    buffer_path = "checkpoints/replay_buffer.pkl"

    # === Initialize model ===
    model = PolicyValueNet(
        board_size=8
    )  # Update if you need board_size or num_blocks args
    optimizer = torch.optim.Adam(model.parameters())

    start_epoch = 1  # Default if no checkpoint

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        model.to(device)

        start_epoch = checkpoint.get("epoch", 1)
        print(f"Loaded model from epoch: {start_epoch}")
    else:
        print("No checkpoint found. Initializing new model.")
        model.to(device)

    # === Load replay buffer ===
    if os.path.exists(buffer_path):
        replay_buffer = ReplayBuffer.load(buffer_path)
        print(f"Loaded buffer with {len(replay_buffer)} samples")
    else:
        raise FileNotFoundError(
            f"No replay buffer found at {buffer_path}. Cannot start training without data."
        )

    # === Initialize trainer ===
    trainer = AlphaZeroTrainer(
        model=model,
        replay_buffer=replay_buffer,
        config=config,
        device=device,
        start_epoch=start_epoch,
    )

    # === Start training ===
    trainer.train()


if __name__ == "__main__":
    main()
