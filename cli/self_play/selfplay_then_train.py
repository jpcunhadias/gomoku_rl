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

    # Model and evaluator
    model = PolicyValueNet(board_size=15)
    evaluator = NeuralEvaluator(model)

    buffer = ReplayBuffer.load("checkpoints/replay_buffer.pkl")
    print(f"Loaded buffer with {len(buffer)} samples.")

    # Self-play
    runner = SelfPlayRunner(
        game_cls=GomokuBoard,
        mcts_cls=MCTS,
        evaluator=evaluator,
        buffer=buffer,
        num_simulations=800,
        temperature_schedule=lambda move: 1.0 if move < 10 else 1e-3,
        verbose=False,
    )

    for i in range(50):
        print(f"→ Self-play game {i + 1}")
        runner.play_game()

    print(f"\nBuffer filled with {len(buffer)} samples")
    buffer.save("checkpoints/replay_buffer.pkl")
    print("Saved buffer after self-play.")

    # Train
    trainer = AlphaZeroTrainer(model, buffer, config, device)
    trainer.train()


if __name__ == "__main__":
    main()
