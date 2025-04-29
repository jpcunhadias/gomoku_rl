import os
import sys

import torch

from game.player import MCTSPlayer
from mcts.mcts import MCTS
from mcts.neural_evaluator import NeuralEvaluator
from model.policy_value_net import PolicyValueNet
from train.config import get_config
from train.replay_buffer import ReplayBuffer
from train.self_play import SelfPlayRunner

# Colab only – adjust this if mounting Google Drive
# from google.colab import drive
# drive.mount('/content/drive')

# === Path setup for Colab environment ===
os.chdir("/content/gomoku_rl")
sys.path.append(".")

# === Load config ===
config = get_config()

# === Prepare buffer save path ===
buffer_path = "/content/drive/MyDrive/gomoku_data/replay_buffer.pkl"
os.makedirs(os.path.dirname(buffer_path), exist_ok=True)

# === Initialize model and evaluator ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model = PolicyValueNet(board_size=15).to(device)
evaluator = NeuralEvaluator(model, device)

# === Create players ===
mcts1 = MCTS(evaluator_fn=evaluator, n_simulations=config.self_play_num_simulations)
mcts2 = MCTS(evaluator_fn=evaluator, n_simulations=config.self_play_num_simulations)

player1 = MCTSPlayer(mcts1, temperature=1.0, add_dirichlet_noise=True)
player2 = MCTSPlayer(mcts2, temperature=1.0)

# === Initialize replay buffer ===
buffer = ReplayBuffer(max_size=config.replay_buffer_size)

# === Self-play runner ===
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

# === Save buffer ===
buffer.save(buffer_path)
print(f"Replay buffer saved to {buffer_path}")
