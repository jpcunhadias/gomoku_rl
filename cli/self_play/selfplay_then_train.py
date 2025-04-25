import torch

from game.gomoku import GomokuBoard
from mcts.mcts import MCTS
from mcts.neural_evaluator import NeuralEvaluator
from model.policy_value_net import PolicyValueNet
from train.config import get_config
from train.replay_buffer import ReplayBuffer
from train.self_play import SelfPlayRunner
from train.train_loop import AlphaZeroTrainer

config = get_config()


def main():
    print("Self-play + training script started.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # === Load model with trained value head ===
    model = PolicyValueNet(board_size=15)
    checkpoint_path = "checkpoints/policy_value_net_epochbest.pth"

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    print(f"Loaded model from checkpoint: {checkpoint_path}")

    evaluator = NeuralEvaluator(model, device)

    # === Load or initialize buffer ===
    buffer_path = "checkpoints/replay_buffer.pkl"
    try:
        buffer = ReplayBuffer.load(buffer_path)
        print(f"Loaded buffer with {len(buffer)} samples.")
    except FileNotFoundError:
        print("No existing buffer found. Creating new buffer.")
        buffer = ReplayBuffer(max_size=config.replay_buffer_size)

    # === Self-play ===
    runner = SelfPlayRunner(
        game_cls=GomokuBoard,
        mcts_cls=MCTS,
        evaluator=evaluator,
        buffer=buffer,
        num_simulations=800,
        temperature_schedule=lambda move: 1.0 if move < 10 else 1e-3,
        verbose=False,
    )

    for i in range(config["num_self_play_games"]):
        print(f"→ Self-play game {i + 1}")
        runner.play_game()

    print(f"\nBuffer filled with {len(buffer)} samples")
    buffer.save(buffer_path)
    print(f"Replay buffer saved to: {buffer_path}")

    # === Training ===
    trainer = AlphaZeroTrainer(model, buffer, config, device)
    trainer.train()


if __name__ == "__main__":
    main()
