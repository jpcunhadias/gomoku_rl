# cli/train/supervised_value_train.py

import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from model.policy_value_net import PolicyValueNet

# === Load Replay Buffer ===
with open("checkpoints/replay_buffer.pkl", "rb") as f:
    buffer = pickle.load(f)

# === Prepare Dataset ===
states, _, z = buffer.sample(len(buffer))

X = states
y = z

dataset = TensorDataset(X, y)
data_loader = DataLoader(dataset, batch_size=128, shuffle=True)

# === Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PolicyValueNet(board_size=8).to(device)
model._init_weights()
# Reinitialize value head with small gain
nn.init.xavier_uniform_(model.value_fc.weight, gain=0.1)
nn.init.zeros_(model.value_fc.bias)

# === Training Setup ===
criterion = nn.SmoothL1Loss(beta=1.0)

base_lr = 1e-3
value_params = list(model.value_conv.parameters()) + list(model.value_fc.parameters())
value_param_ids = {id(p) for p in value_params}
policy_params = [p for p in model.parameters() if id(p) not in value_param_ids]

optimizer = optim.Adam(
    [
        {"params": policy_params, "lr": base_lr},
        {
            "params": value_params,
            "lr": base_lr * 0.3,
            "weight_decay": 2e-4,
        },
    ]
)

loss_history = []
mae_history = []

# === Training Loop ===
for epoch in range(1, 11):
    model.train()
    total_loss = 0
    total_mae = 0
    total = 0

    for batch_x, batch_y in data_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        _, value_pred = model(batch_x)

        loss = criterion(value_pred.view(-1), batch_y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        total_mae += torch.abs(value_pred.view(-1) - batch_y).sum().item()
        total += batch_y.size(0)

    avg_loss = total_loss / total
    avg_mae = total_mae / total

    loss_history.append(avg_loss)
    mae_history.append(avg_mae)
    print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, MAE = {avg_mae:.4f}")

# === Plot ===
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(loss_history, label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(mae_history, label="MAE")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.title("Mean Absolute Error")
plt.grid(True)

plt.tight_layout()
plt.savefig("value_head_supervised_training.png")
plt.show()
