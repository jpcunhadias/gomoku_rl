from game.gomoku import GomokuBoard
from mcts.mcts import MCTS
from mcts.neural_evaluator import NeuralEvaluator
from model.policy_value_net import PolicyValueNet
from train.replay_buffer import ReplayBuffer
from train.self_play import SelfPlayRunner

if __name__ == "__main__":
    # Setup model and evaluator
    model = PolicyValueNet(board_size=15)  # adapt if needed
    evaluator = NeuralEvaluator(model)

    # Setup replay buffer
    buffer = ReplayBuffer(max_size=100)

    # Setup self-play runner
    runner = SelfPlayRunner(
        game_cls=GomokuBoard,
        mcts_cls=MCTS,
        evaluator=evaluator,
        buffer=buffer,
        num_simulations=50,  # for fast test
        temperature_schedule=lambda move: 1.0 if move < 10 else 1e-3,
        verbose=True,  # Set to True for detailed output
    )

    print("Running one self-play game...")
    runner.play_game()

    print(f"Game finished. Buffer has {len(buffer)} samples.")
    samples = buffer.sample(1)
    s, pi, z = samples

    print("Sample state shape:", s.shape)  # Expect: [1, 3, 15, 15]
    print("Sample π shape:", pi.shape)  # Expect: [1, 15, 15]
    print("Sample z:", z.item())  # Expect: -1, 0, or 1

    # buffer.save("checkpoints/replay_buffer.pkl")
    # print("Replay buffer saved to checkpoints/replay_buffer.pkl")
