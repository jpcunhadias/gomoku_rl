import os
import sys

from train.config import get_config
from train.self_play import run_selfplay

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

if __name__ == "__main__":
    run_selfplay(config, buffer_save_path=buffer_path)
