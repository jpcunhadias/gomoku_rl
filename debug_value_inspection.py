import pickle
import torch
from model.policy_value_net import PolicyValueNet
from train.replay_buffer import ReplayBuffer

# Initialize the same architecture used for training
model = PolicyValueNet(board_size=15)

# Load the trained weights from your checkpoint
checkpoint = torch.load("checkpoints/debug_model_epoch10.pth", map_location='cpu')
model.load_state_dict(checkpoint["model_state_dict"])
print(f"Loaded model from epoch: {checkpoint['epoch']}")
model.eval()  # set to inference mode

# Load the replay buffer as a list of samples
with open("checkpoints/replay_buffer.pkl", "rb") as f:
    data_list = pickle.load(f)

# Re-wrap as a ReplayBuffer
buffer = ReplayBuffer(max_size=len(data_list))
buffer.buffer = data_list

# Sample 8 examples from buffer
states, _, target_values = buffer.sample(batch_size=8)

# Forward pass through the model
with torch.no_grad():
    _, value_pred = model(states)

print("Target z (actual game outcomes):")
print(target_values.squeeze().tolist())

print("\nPredicted v (value head outputs):")
print(value_pred.squeeze().tolist())
