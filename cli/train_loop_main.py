import torch
from model.policy_value_net import PolicyValueNet
from train.replay_buffer import ReplayBuffer
from train.train_loop import AlphaZeroTrainer
from train.config import config


def main():
    print("Starting AlphaZero training loop")

    # Select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize model
    model = PolicyValueNet()  # update if your net requires more args

    # Load or create replay buffer
    replay_buffer = ReplayBuffer(config.replay_buffer_size)

    # Initialize trainer
    trainer = AlphaZeroTrainer(model, replay_buffer, config, device=device)

    # Start training
    trainer.train()


if __name__ == '__main__':
    main()
