# debug_value_inspection.py

import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from model.policy_value_net import PolicyValueNet


def main():
    # === Load Model ===
    model = PolicyValueNet(board_size=15)
    checkpoint = torch.load(
        "checkpoints/policy_value_net_best.pth",
        map_location="cuda" if torch.cuda.is_available() else "cpu",
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded model from epoch: {checkpoint['epoch']}")
    model.eval()

    # === Load Replay Buffer ===
    with open("checkpoints/replay_buffer.pkl", "rb") as f:
        buffer = pickle.load(f)

    print(f"Loaded buffer: {type(buffer)}")

    # === Sample data ===
    states, _, target_values = buffer.sample(batch_size=256)

    with torch.no_grad():
        _, value_pred = model(states)

    z = target_values.view(-1).cpu().numpy()
    v = value_pred.view(-1).cpu().numpy()

    print(f"Length of z (target): {len(z)}")
    print(f"Length of v (predicted): {len(v)}")

    # === Basic Stats ===
    v_for_z_pos = v[z == 1.0]
    v_for_z_neg = v[z == -1.0]

    print("\nFor z = +1 (win):")
    print(f"  Mean: {v_for_z_pos.mean():.4f}, Std: {v_for_z_pos.std():.4f}")

    print("\nFor z = -1 (loss):")
    print(f"  Mean: {v_for_z_neg.mean():.4f}, Std: {v_for_z_neg.std():.4f}")

    predicted_class = np.where(v >= 0, 1.0, -1.0)
    accuracy = np.mean(predicted_class == z)
    print(f"\nBinary prediction accuracy (v >= 0 means win): {accuracy:.2%}")

    # === Plot: Histogram
    plt.figure(figsize=(8, 5))
    plt.hist(v_for_z_pos, bins=30, alpha=0.6, label="z = +1 (win)")
    plt.hist(v_for_z_neg, bins=30, alpha=0.6, label="z = -1 (loss)")
    plt.title("Value Head Output Distribution by Game Outcome (z)")
    plt.xlabel("Predicted v")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("value_pred_by_z_histogram.png")
    plt.show()

    # === Plot: Scatter
    plt.figure(figsize=(8, 5))
    plt.scatter(z, v, alpha=0.6, edgecolors="k")
    plt.xlabel("Target z (Game Outcome)")
    plt.ylabel("Predicted v (Value Head Output)")
    plt.title("Scatter: Predicted v vs. True z")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("scatter_z_vs_v.png")
    plt.show()

    # === Plot: PCA of value features
    features = model.extract_value_features(
        states
    )  # Ensure your model has this method!
    proj = PCA(n_components=2).fit_transform(features.detach().cpu().numpy())
    colors = ["tab:blue" if label == 1.0 else "tab:orange" for label in z]

    plt.figure(figsize=(8, 6))
    plt.scatter(proj[:, 0], proj[:, 1], c=colors, alpha=0.6, edgecolors="k")
    plt.title("PCA of Value Head Features")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("pca_value_features_by_z.png")
    plt.show()


if __name__ == "__main__":
    main()
