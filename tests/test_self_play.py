import pytest
from game.player import MCTSPlayer
from mcts.mcts import MCTS
from mcts.neural_evaluator import NeuralEvaluator
from model.policy_value_net import PolicyValueNet
from train.replay_buffer import ReplayBuffer
from train.self_play import SelfPlayRunner


def test_self_play_game_populates_buffer():
    model = PolicyValueNet(board_size=8)
    evaluator = NeuralEvaluator(model)
    buffer = ReplayBuffer(max_size=100)

    mcts_player1 = MCTSPlayer(
        MCTS(evaluator_fn=evaluator, n_simulations=50),
        temperature=1.0,
    )
    mcts_player2 = MCTSPlayer(
        MCTS(evaluator_fn=evaluator, n_simulations=50), temperature=1.0
    )

    runner = SelfPlayRunner(
        player1=mcts_player1,
        player2=mcts_player2,
        buffer=buffer,
        temperature_schedule=lambda move: 1.0 if move < 10 else 1e-3,
        verbose=False,
    )

    runner.play_game()
    assert len(buffer) > 0, "Replay buffer is empty after self-play!"

    states, policies, values = buffer.sample(1)
    assert states[0].shape == (3, 8, 8)
    assert policies[0].shape == (8, 8)
    assert values[0].item() in [0, 1, 2]


if __name__ == "__main__":
    pytest.main()
