import os

from model.policy_value_net import PolicyValueNet
from train.config import get_config
from train.replay_buffer import ReplayBuffer
from train.train_loop import run_training

# Optional: mount Google Drive if needed
# try:
#     from google.colab import drive
#
#     drive.mount("/content/drive")
# except ImportError:
#     pass

# Load config
config = get_config()

buffer_path = "/content/drive/MyDrive/gomoku_data/replay_buffer.pkl"

resume_training = False  # Set to False if you want a fresh model
checkpoint_path = "/content/drive/MyDrive/gomoku_checkpoints/policy_value_net.pth"

config.save_path = checkpoint_path

print("Loading replay buffer...")
buffer = ReplayBuffer.load(buffer_path)
print(f"Replay buffer loaded with {len(buffer)} samples.")
print(f"Replay buffer loaded with {len(buffer)} samples.")

# --- Model initialization
if resume_training and os.path.exists(checkpoint_path):
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = PolicyValueNet.load_from_checkpoint(checkpoint_path, board_size=15)
else:
    print("Initializing new model from scratch.")
    model = PolicyValueNet(board_size=15)

# Run training
print("Starting training...")
run_training(model, buffer, config)
