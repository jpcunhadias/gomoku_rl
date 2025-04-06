import torch
from tqdm import trange
from game.gomoku import GomokuBoard
from mcts.mcts import MCTS
from mcts.neural_evaluator import NeuralEvaluator
from model.policy_value_net import PolicyValueNet
from train.config import config
from train.replay_buffer import ReplayBuffer
from train.self_play import SelfPlayRunner
from train.train_loop import AlphaZeroTrainer


def run_selfplay(num_games=50, mcts_simulations=800, buffer_save_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Model and evaluator
    model = PolicyValueNet(board_size=15)
    evaluator = NeuralEvaluator(model)

    # Shared buffer
    buffer = ReplayBuffer(max_size=config.replay_buffer_size)

    # Self-play
    runner = SelfPlayRunner(
        game_cls=GomokuBoard,
        mcts_cls=MCTS,
        evaluator=evaluator,
        buffer=buffer,
        num_simulations=mcts_simulations,
        temperature_schedule=lambda move: 1.0 if move < 10 else 1e-3,
        verbose=False,
    )

    for i in trange(num_games, desc="Self-play games"):
        runner.play_game()

    print(f"\nBuffer filled with {len(buffer)} samples")

    if buffer_save_path:
        buffer.save(buffer_save_path)

    return model, buffer


def run_training(model, buffer, save_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    trainer = AlphaZeroTrainer(model, buffer, config, device)
    trainer.train()

    if save_path:
        torch.save(model.state_dict(), save_path)


def main():
    print("Self-play + training (CPU test)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Model and evaluator
    model = PolicyValueNet(board_size=15)
    evaluator = NeuralEvaluator(model)

    # Shared buffer
    buffer = ReplayBuffer(max_size=config.replay_buffer_size)

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

    # Train
    trainer = AlphaZeroTrainer(model, buffer, config, device)
    trainer.train()


if __name__ == "__main__":
    main()
