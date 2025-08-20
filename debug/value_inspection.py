import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from model.policy_value_net import PolicyValueNet
from train.replay_buffer import ReplayBuffer

CHECKPOINT_PATH = "checkpoints/policy_value_net_best.pth"
BUFFER_PATH = "checkpoints/replay_buffer.pkl"
OUTPUT_DIR = "debug/debug_outputs"


def plot_histograms(v: np.ndarray, z: np.ndarray) -> None:
    v_pos = v[z == 1.0]
    v_neg = v[z == -1.0]
    v_draw = v[z == 0.0]

    plt.figure(figsize=(8, 5))
    plt.hist(v_pos, bins=30, alpha=0.6, label="z = +1 (win)")
    plt.hist(v_neg, bins=30, alpha=0.6, label="z = -1 (loss)")
    plt.hist(v_draw, bins=30, alpha=0.6, label="z =  0 (draw)")
    plt.title("Value Head Output Distribution by Game Outcome (z)")
    plt.xlabel("Predicted v")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "value_pred_by_z_histogram.png"))
    plt.show()


def plot_scatter(z: np.ndarray, v: np.ndarray) -> None:
    plt.figure(figsize=(8, 5))
    plt.scatter(z, v, alpha=0.6, edgecolors="k")
    plt.xlabel("Target z (Game Outcome)")
    plt.ylabel("Predicted v (Value Head Output)")
    plt.title("Scatter: Predicted v vs. True z")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "scatter_z_vs_v.png"))
    plt.show()


def plot_pca(features: torch.Tensor, z: np.ndarray) -> None:
    proj = PCA(n_components=2).fit_transform(features.detach().cpu().numpy())
    color_map = {
        -1.0: "tab:orange",  # loss
        0.0: "tab:gray",  # draw
        1.0: "tab:blue",  # win
    }
    colors = [color_map.get(label, "black") for label in z]

    plt.figure(figsize=(8, 6))
    plt.scatter(proj[:, 0], proj[:, 1], c=colors, alpha=0.6, edgecolors="k")
    plt.title("PCA of Value Head Features")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pca_value_features_by_z.png"))
    plt.show()


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # === Load model and buffer
    model = PolicyValueNet.load_from_checkpoint(
        path=CHECKPOINT_PATH,
        board_size=8,
        device=device,
    )
    buffer = ReplayBuffer.load(BUFFER_PATH)
    print(f"Loaded model from checkpoint: {CHECKPOINT_PATH}")
    print(f"Loaded buffer with {len(buffer)} samples")

    # === Stratified sampling: ensure all classes {-1, 0, +1} present
    MAX_TRIES = 10
    for _ in range(MAX_TRIES):
        states, _, target_values = buffer.sample(batch_size=256)
        z = target_values.view(-1).cpu().numpy() - 1  # Convert to {-1, 0, +1}
        if {-1.0, 0.0, 1.0}.issubset(set(z)):
            break
    else:
        raise RuntimeError("Could not sample all 3 classes after 10 attempts.")

    states = states.to(device)
    target_values = target_values.to(device)

    with torch.no_grad():
        _, value_pred = model(states)  # shape (B, 3)

    print("\nSample raw value_pred logits (first 5 samples):")
    print(value_pred[:5].cpu().numpy())
    print("value_pred shape:", value_pred.shape)

    # Convert logits to scalar value in [-1, 1]
    if value_pred.ndim == 2 and value_pred.shape[1] == 3:
        probs = torch.softmax(value_pred, dim=1)  # (B, 3)
        v_tensor = probs[:, 2] - probs[:, 0]  # P(win) - P(loss)
    else:
        raise ValueError(f"Expected shape (B, 3), got {value_pred.shape}")

    v = v_tensor.cpu().numpy()
    z = target_values.view(-1).cpu().numpy() - 1  # Match analysis with label format

    print(
        f"Sampled {len(z)} values. Positive z: {(z == 1.0).sum()}, "
        f"Negative z: {(z == -1.0).sum()}, Draws: {(z == 0.0).sum()}"
    )

    # === Stats
    v_pos = v[z == 1.0]
    v_neg = v[z == -1.0]
    v_draw = v[z == 0.0]

    print(f"\nFor z = +1 (win):  Mean = {v_pos.mean():.4f}, Std = {v_pos.std():.4f}")
    print(f"For z = -1 (loss): Mean = {v_neg.mean():.4f}, Std = {v_neg.std():.4f}")
    print(f"For z =  0 (draw):  Mean = {v_draw.mean():.4f}, Std = {v_draw.std():.4f}")

    # 3-class accuracy: argmax over logits → map to {-1, 0, +1}
    z_pred = torch.argmax(probs, dim=1).cpu().numpy() - 1
    acc = (z_pred == z).mean()
    print(f"\n3-class prediction accuracy (argmax): {acc:.2%}")

    # === Plots
    plot_histograms(v, z)
    plot_scatter(z, v)

    value_features = model.extract_value_features(states)
    plot_pca(value_features, z)


if __name__ == "__main__":
    main()
