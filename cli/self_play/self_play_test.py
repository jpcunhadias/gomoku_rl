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
    assert len(buffer) > 0, "Replay buffer is empty after self-play!"

    print(f"Game finished. Buffer has {len(buffer)} samples.")

    states, policies, values = buffer.sample(1)

    print("Sample state shape:", states[0].shape)  # Expect: [3, 15, 15]
    print("Sample π shape:", policies[0].shape)     # Expect: [15, 15]
    print("Sample z:", values[0].item())             # Expect: -1, 0, or 1

    # Optional save
    # buffer.save("checkpoints/replay_buffer.pkl")
    # print("Replay buffer saved to checkpoints/replay_buffer.pkl")
