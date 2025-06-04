import torch

from game.player import MCTSPlayer
from mcts.mcts import MCTS
from mcts.neural_evaluator import NeuralEvaluator
from model.policy_value_net import PolicyValueNet
from train.config import get_config
from train.replay_buffer import ReplayBuffer
from train.self_play import SelfPlayRunner
from train.train_loop import AlphaZeroTrainer

config = get_config()


def main() -> None:
    """Run a short cycle of self-play followed by training."""
    print("Self-play + training script started.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # === Initialize NEW model ===
    model = PolicyValueNet(board_size=15).to(device)
    print("Initialized new PolicyValueNet.")

    evaluator = NeuralEvaluator(model, device)

    # === Initialize NEW buffer ===
    buffer = ReplayBuffer(max_size=config.replay_buffer_size)
    print("Initialized new ReplayBuffer.")

    # === Create MCTS Players ===
    player1 = MCTSPlayer(
        mcts=MCTS(
            evaluator_fn=evaluator, n_simulations=config.self_play_num_simulations
        ),
        temperature=1.0,  # initial temp
        add_dirichlet_noise=True,  # enable noise on first move
    )
    player2 = MCTSPlayer(
        mcts=MCTS(
            evaluator_fn=evaluator, n_simulations=config.self_play_num_simulations
        ),
        temperature=1.0,
        add_dirichlet_noise=True,
    )

    # === Self-play ===
    runner = SelfPlayRunner(
        player1=player1,
        player2=player2,
        buffer=buffer,
        temperature_schedule=lambda move: 1.0 if move < 10 else 1e-3,
        verbose=False,
    )

    for i in range(config.num_self_play_games):
        print(f"→ Self-play game {i + 1}")
        runner.play_game()

    print(f"\nBuffer filled with {len(buffer)} samples")

    buffer.save("checkpoints/replay_buffer.pkl")
    print("Replay buffer saved to: checkpoints/replay_buffer.pkl")

    # === Training ===
    trainer = AlphaZeroTrainer(model, buffer, config, device)
    trainer.train()


if __name__ == "__main__":
    main()
