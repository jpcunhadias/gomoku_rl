import os

from model.policy_value_net import PolicyValueNet
from train.config import get_config
from train.replay_buffer import ReplayBuffer
from train.train_loop import run_training

# === Optional: mount Google Drive ===
# from google.colab import drive
# drive.mount("/content/drive")

# === Load config ===
config = get_config()

# === Paths ===
buffer_path = "/content/drive/MyDrive/gomoku_data/replay_buffer.pkl"
checkpoint_path = "/content/drive/MyDrive/gomoku_checkpoints/policy_value_net.pth"
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

resume_training = False
config.save_path = checkpoint_path

# === Load Replay Buffer ===
print("Loading replay buffer...")
buffer = ReplayBuffer.load(buffer_path)
print(f"Replay buffer loaded with {len(buffer)} samples.")

# === Initialize model ===
if resume_training and os.path.exists(checkpoint_path):
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = PolicyValueNet.load_from_checkpoint(checkpoint_path, board_size=8)
else:
    print("Initializing new model from scratch.")
    model = PolicyValueNet(board_size=8)

# === Start training ===
print("Starting training...")
run_training(model, buffer, config)
