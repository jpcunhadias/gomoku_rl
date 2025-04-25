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

y = torch.tensor(
    [1.0 if val == 1.0 else 0.0 for val in z.view(-1)], dtype=torch.float32
).unsqueeze(1)

dataset = TensorDataset(X, y)
data_loader = DataLoader(dataset, batch_size=128, shuffle=True)

# === Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PolicyValueNet(board_size=15).to(device)

# === Training Setup ===
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

loss_history = []
accuracy_history = []

# === Training Loop ===
for epoch in range(1, 11):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in data_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        _, value_pred = model(batch_x)

        loss = criterion(value_pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)

        pred_binary = (torch.sigmoid(value_pred) >= 0.5).float()
        correct += (pred_binary == batch_y).sum().item()
        total += batch_y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    loss_history.append(avg_loss)
    accuracy_history.append(accuracy)
    print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2%}")

# === Plot ===
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(loss_history, label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(accuracy_history, label="Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")
plt.grid(True)

plt.tight_layout()
plt.savefig("value_head_supervised_training.png")
plt.show()
