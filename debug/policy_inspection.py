import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.nn import functional as F

from model.policy_value_net import PolicyValueNet
from train.replay_buffer import ReplayBuffer

CHECKPOINT_PATH = "checkpoints/policy_value_net_best.pth"
BUFFER_PATH = "checkpoints/replay_buffer.pkl"
OUTPUT_DIR = "debug/debug_outputs"


def compute_kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    p = p + 1e-8  # Avoid log(0)
    q = q + 1e-8
    return (p * (p.log() - q.log())).sum(dim=1)


def plot_entropy_histogram(entropy: np.ndarray) -> None:
    plt.figure(figsize=(7, 5))
    plt.hist(entropy, bins=40, alpha=0.7, color="tab:blue")
    plt.xlabel("Policy Entropy")
    plt.ylabel("Frequency")
    plt.title("Histogram of Policy Entropy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "policy_entropy_histogram.png"))
    plt.show()


def plot_kl_histogram(kl_div: np.ndarray) -> None:
    plt.figure(figsize=(7, 5))
    plt.hist(kl_div, bins=40, alpha=0.7, color="tab:red")
    plt.xlabel("KL Divergence (Model || MCTS Target)")
    plt.ylabel("Frequency")
    plt.title("Histogram of KL Divergence")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "kl_divergence_histogram.png"))
    plt.show()


def plot_entropy_kl_scatter(entropy: np.ndarray, kl_div: np.ndarray) -> None:
    plt.figure(figsize=(7, 5))
    plt.scatter(entropy, kl_div, alpha=0.6, edgecolors="k")
    plt.xlabel("Entropy")
    plt.ylabel("KL Divergence")
    plt.title("Entropy vs KL Divergence per Sample")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "entropy_vs_kl_scatter.png"))
    plt.show()


def plot_pca(features: torch.Tensor) -> None:
    proj = PCA(n_components=2).fit_transform(features.detach().cpu().numpy())
    plt.figure(figsize=(8, 6))
    plt.scatter(proj[:, 0], proj[:, 1], alpha=0.6, edgecolors="k")
    plt.title("PCA of Policy Head Features")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pca_policy_features_by_move.png"))
    plt.show()


def topk_accuracy(preds: torch.Tensor, targets: torch.Tensor, k: int = 3) -> float:
    topk = torch.topk(preds, k, dim=1).indices
    best_moves = torch.argmax(targets, dim=1, keepdim=True)
    correct = topk.eq(best_moves).any(dim=1).float()
    return correct.mean().item()


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = PolicyValueNet.load_from_checkpoint(
        path=CHECKPOINT_PATH,
        board_size=8,
        device=device,
    )
    buffer = ReplayBuffer.load(BUFFER_PATH)

    states, target_policies, _ = buffer.sample(batch_size=256)
    states = states.to(device)
    target_policies = target_policies.to(device)

    # Flatten target_policies if 3D (e.g., [B, 8, 8]) → [B, 64]
    if target_policies.ndim == 3:
        target_policies = target_policies.view(target_policies.size(0), -1)

    with torch.no_grad():
        policy_logits, _ = model(states)
        probs = F.softmax(policy_logits, dim=1)
        entropy = -(probs * probs.log()).sum(dim=1).cpu().numpy()

        assert probs.shape == target_policies.shape, (
            f"Shape mismatch: probs={probs.shape}, targets={target_policies.shape}"
        )

        kl_div = compute_kl_divergence(probs, target_policies).cpu().numpy()

    # === Stats ===
    acc = topk_accuracy(probs, target_policies, k=3)
    print(f"Top-3 accuracy against MCTS target: {acc:.2%}")

    # === Sample comparisons
    print("\nSample comparison of predicted vs target moves:")
    for i in range(5):
        pred_top = torch.topk(probs[i], 5).indices.cpu().numpy()
        target_top = torch.topk(target_policies[i], 5).indices.cpu().numpy()
        print(f"Sample {i}: Pred Top-5 = {pred_top}, Target Top-5 = {target_top}")

    # === Save summary
    with open(os.path.join(OUTPUT_DIR, "policy_debug_summary.txt"), "w") as f:
        f.write(f"Top-3 accuracy: {acc:.4f}\n")
        f.write(f"Entropy mean: {entropy.mean():.4f}, std: {entropy.std():.4f}\n")
        f.write(f"KL mean: {kl_div.mean():.4f}, std: {kl_div.std():.4f}\n")

    # === Plots
    plot_entropy_histogram(entropy)
    plot_kl_histogram(kl_div)
    plot_entropy_kl_scatter(entropy, kl_div)

    # === Optional: Feature space visualization
    if hasattr(model, "extract_policy_features"):
        features = model.extract_policy_features(states)
        plot_pca(features)
    else:
        print("[Skip] Model does not support extract_policy_features().")


if __name__ == "__main__":
    main()
