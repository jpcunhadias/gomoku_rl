from model.policy_value_net import PolicyValueNet
from train.config import get_config
from train.replay_buffer import ReplayBuffer
from train.train_loop import run_training

# Optional: mount Google Drive if needed
try:
    from google.colab import drive

    drive.mount("/content/drive")
except ImportError:
    pass

# Load config
config = get_config()

# Define buffer path (matches where you saved it in Colab)
buffer_path = "/content/drive/MyDrive/gomoku_data/replay_buffer.pkl"

# Load replay buffer from file
print("Loading replay buffer...")
buffer = ReplayBuffer.load_from_file(buffer_path, max_size=config.replay_buffer_size)
print(f"Replay buffer loaded with {len(buffer)} samples.")

# Initialize a new model (or replace this line with load_from_checkpoint if needed)
model = PolicyValueNet(board_size=15)

# Run training
print("Starting training...")
run_training(model, buffer, config)
