import os
import sys
from train.config import get_config
from train.self_play import run_selfplay

# Mount Google Drive (Colab only)
# try:
#     from google.colab import drive
#
#     drive.mount("/content/drive")
# except ImportError:
#     pass

os.chdir("/content/gomoku_rl")
sys.path.append(".")
# Load config
config = get_config()

# Prepare paths
os.makedirs(os.path.dirname(config.save_path), exist_ok=True)
buffer_path = "/content/drive/MyDrive/gomoku_data/replay_buffer.pkl"
os.makedirs(os.path.dirname(buffer_path), exist_ok=True)

model, buffer = run_selfplay(
    config=config,
    num_games=config.num_self_play_games,
    mcts_simulations=config.num_simulations,
    buffer_save_path=buffer_path,
)

print(f"Replay buffer saved to {buffer_path}")
