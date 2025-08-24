#!/usr/bin/env python3
# debug/value_head_check.py
"""
Checks the scalar value head end-to-end on a sample from the replay buffer.

It verifies:
- Shapes & ranges (value in [-1, 1])
- Target dtype/values (floats in {-1,0,+1} or legacy {0,1,2} remapped)
- Brier score (regression)
- Bin-based calibration (ECE-style for scalar)
- Pre-tanh saturation (% |pre_tanh| > 2.0), via a forward hook on value_fc
- Threshold sanity for wins/losses/draws
- Plots: histograms by target class, scatter z vs v̂, optional PCA of value features

Usage:
  python debug/value_head_check.py \
      --checkpoint checkpoints/policy_value_net_best.pth \
      --buffer checkpoints/replay_buffer.pkl \
      --batch 256 \
      --output debug/debug_outputs
"""

import argparse
import math
import os
import random
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

# Optional: PCA of value features (only if model exposes extract_value_features)
try:
    from sklearn.decomposition import PCA  # type: ignore

    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# Project imports (adapt paths if your layout differs)
from model.policy_value_net import PolicyValueNet
from train.replay_buffer import ReplayBuffer


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(checkpoint: str, device: str, board_size: int = 8) -> PolicyValueNet:
    model = PolicyValueNet.load_from_checkpoint(
        path=checkpoint, board_size=board_size, device=device
    )
    model.eval()
    return model


def load_buffer(path: str) -> ReplayBuffer:
    return ReplayBuffer.load(path)


def sample_batch_with_all_classes(
    buffer: ReplayBuffer, batch_size: int, tries: int = 10
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Try to get a batch that contains all classes {-1,0,+1} (or {0,1,2} legacy).
    Falls back to the last sample if not possible (small buffers).
    """
    last = None
    for _ in range(tries):
        states, policies, values = buffer.sample(batch_size)
        last = (states, policies, values)
        vals = values.detach().cpu().numpy().astype(float)
        unique = set(np.unique(vals).tolist())
        if unique.issuperset({-1.0, 0.0, 1.0}) or unique.issuperset({0.0, 1.0, 2.0}):
            return states, policies, values
    # Fallback: return whatever we got last time
    return last  # type: ignore


def maybe_map_legacy_targets_to_scalar(values: torch.Tensor) -> torch.Tensor:
    """
    If targets look like {0,1,2}, map to {-1,0,+1}. Ensure float32 dtype.
    """
    vals = values.detach().cpu().numpy()
    uniq = np.unique(vals)
    if set(uniq.tolist()).issubset({0, 1, 2}):
        mapped = (vals - 1.0).astype(np.float32)
        return torch.from_numpy(mapped)
    # Already scalar {-1,0,+1} or close; cast to float32
    return values.to(dtype=torch.float32).cpu()


def brier_score(v_hat: np.ndarray, z: np.ndarray) -> float:
    return float(np.mean((v_hat - z) ** 2))


def scalar_ece(
    v_hat: np.ndarray, z: np.ndarray, num_bins: int = 10
) -> Tuple[float, np.ndarray]:
    """
    ECE-style metric for scalar [-1,1]: partition predictions into bins, compute
    |mean_pred - mean_true| per bin, and return weighted average by bin mass.
    """
    bins = np.linspace(-1.0, 1.0, num_bins + 1)
    indices = np.digitize(v_hat, bins) - 1  # 0..num_bins-1
    eces = []
    weights = []
    per_bin = []

    for b in range(num_bins):
        mask = indices == b
        cnt = int(mask.sum())
        if cnt == 0:
            per_bin.append((b, cnt, np.nan, np.nan, np.nan))
            continue
        mean_pred = float(v_hat[mask].mean())
        mean_true = float(z[mask].mean())
        gap = abs(mean_pred - mean_true)
        eces.append(gap * (cnt / len(v_hat)))
        weights.append(cnt)
        per_bin.append((b, cnt, mean_pred, mean_true, gap))

    ece = float(np.nansum(eces))
    return ece, np.array(per_bin, dtype=object)


def attach_pretanh_hook(model: PolicyValueNet):
    """
    Attaches a forward hook to model.value_fc to capture pre-tanh activations.
    Returns a callable to fetch and clear the last captured tensor.
    """
    store = {"pre": None}

    # Expect model.value_fc to exist; if not, we skip gracefully
    layer = getattr(model, "value_fc", None)

    def hook(_module, input, output):
        # output is pre-tanh if hook is on value_fc; flatten batch dimension
        store["pre"] = output.detach().view(-1).cpu()

    handle = None
    if layer is not None:
        handle = layer.register_forward_hook(hook)

    def fetch_and_clear():
        t = store["pre"]
        store["pre"] = None
        return t

    return handle, fetch_and_clear


def plot_histograms(v_hat: np.ndarray, z: np.ndarray, outdir: str) -> None:
    v_pos = v_hat[z == 1.0]
    v_neg = v_hat[z == -1.0]
    v_draw = v_hat[z == 0.0]

    plt.figure(figsize=(8, 5))
    plt.hist(v_pos, bins=30, alpha=0.6, label="z = +1 (win)")
    plt.hist(v_neg, bins=30, alpha=0.6, label="z = -1 (loss)")
    plt.hist(v_draw, bins=30, alpha=0.6, label="z =  0 (draw)")
    plt.title("Value Head Output Distribution by Outcome z")
    plt.xlabel("Predicted v̂")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "value_pred_by_z_histogram.png"))
    plt.close()


def plot_scatter(z: np.ndarray, v_hat: np.ndarray, outdir: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.scatter(z, v_hat, alpha=0.6, edgecolors="k")
    plt.xlabel("Target z")
    plt.ylabel("Predicted v̂")
    plt.title("Scatter: v̂ vs z")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "scatter_z_vs_v.png"))
    plt.close()


def maybe_plot_pca_value_features(
    model: PolicyValueNet, states: torch.Tensor, z: np.ndarray, outdir: str
) -> None:
    if not SKLEARN_OK:
        print("[INFO] sklearn not available; skipping PCA plot.")
        return
    if not hasattr(model, "extract_value_features"):
        print("[INFO] model.extract_value_features not found; skipping PCA plot.")
        return
    with torch.no_grad():
        feats = model.extract_value_features(states)  # expect [B, D]
        if feats.dim() > 2:
            feats = feats.view(feats.size(0), -1)
        X = feats.detach().cpu().numpy()
        proj = PCA(n_components=2).fit_transform(X)
    color_map = {-1.0: "tab:orange", 0.0: "tab:gray", 1.0: "tab:blue"}
    colors = [color_map.get(float(lbl), "black") for lbl in z]

    plt.figure(figsize=(8, 6))
    plt.scatter(proj[:, 0], proj[:, 1], c=colors, alpha=0.6, edgecolors="k")
    plt.title("PCA of Value Features (colored by z)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pca_value_features_by_z.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--buffer", type=str, required=True)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--board_size", type=int, default=8)
    parser.add_argument("--output", type=str, default="debug/debug_outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seeds(args.seed)
    print(f"[INFO] device={device}, seed={args.seed}")

    # Load artifacts
    model = load_model(args.checkpoint, device=device, board_size=args.board_size)
    buffer = load_buffer(args.buffer)
    print(f"[INFO] Loaded model: {args.checkpoint}")
    print(f"[INFO] Loaded buffer with {len(buffer)} samples")

    # Attach pre-tanh hook (if available)
    hook_handle, fetch_pretanh = attach_pretanh_hook(model)

    # Sample
    states, _, values_raw = sample_batch_with_all_classes(buffer, args.batch)
    z = maybe_map_legacy_targets_to_scalar(values_raw).numpy().astype(np.float32)
    unique_z, counts_z = np.unique(z, return_counts=True)
    z_counts = {float(k): int(v) for k, v in zip(unique_z.tolist(), counts_z.tolist())}

    # Move states to device & infer
    states = states.to(device)
    with torch.no_grad():
        policy_logits, value_pred = model(states)
    v_hat = value_pred.view(-1).detach().cpu().numpy()

    # Fetch pre-tanh (if hook attached)
    pre = fetch_pretanh()
    pretanh_available = pre is not None
    if pretanh_available:
        pre = pre.numpy()
        sat_share = float(np.mean(np.abs(pre) > 2.0))
    else:
        sat_share = float("nan")

    # Basic sanity
    v_min, v_max = float(np.min(v_hat)), float(np.max(v_hat))
    in_range = (v_min >= -1.0001) and (v_max <= 1.0001)

    print("\n=== VALUE HEAD CHECK ===")
    print(
        f"Pred shape: {value_pred.shape}  |  v̂ range: [{v_min:.4f}, {v_max:.4f}]  |  in [-1,1]? {in_range}"
    )
    print(f"Targets present (counts): {z_counts}")

    # Metrics
    bs = brier_score(v_hat, z)
    ece, per_bin = scalar_ece(v_hat, z, num_bins=10)

    print(f"\nBrier score: {bs:.6f}")
    print(f"Scalar ECE (10 bins): {ece:.6f}")
    print("\nCalibration bins (bin_idx, count, mean_pred, mean_true, abs_gap):")
    for row in per_bin:
        b, cnt, mp, mt, gap = row
        mp_s = "nan" if isinstance(mp, float) and math.isnan(mp) else f"{mp:.4f}"
        mt_s = "nan" if isinstance(mt, float) and math.isnan(mt) else f"{mt:.4f}"
        gap_s = "nan" if isinstance(gap, float) and math.isnan(gap) else f"{gap:.4f}"
        print(f"  {int(b):2d}  {int(cnt):4d}  {mp_s:>8}  {mt_s:>8}  {gap_s:>8}")

    # Threshold sanity
    pos_mask = z == 1.0
    neg_mask = z == -1.0
    draw_mask = z == 0.0
    pos_rate = float(np.mean(v_hat[pos_mask] > 0.7)) if pos_mask.any() else float("nan")
    neg_rate = (
        float(np.mean(v_hat[neg_mask] < -0.7)) if neg_mask.any() else float("nan")
    )
    draw_rate = (
        float(np.mean(np.abs(v_hat[draw_mask]) < 0.2))
        if draw_mask.any()
        else float("nan")
    )

    print("\nThreshold sanity:")
    print(f"  P(v̂>0.7 | z=+1): {pos_rate:.3f}")
    print(f"  P(v̂<-0.7 | z=-1): {neg_rate:.3f}")
    print(f"  P(|v̂|<0.2 | z=0): {draw_rate:.3f}")

    # Pre-tanh saturation
    if pretanh_available:
        print(f"\nPre-tanh saturation: share(|pre_tanh|>2.0) = {sat_share:.3f}")
    else:
        print(
            "\n[WARN] Could not capture pre-tanh activations (no model.value_fc hook)."
        )

    # Plots
    plot_histograms(v_hat, z, args.output)
    plot_scatter(z, v_hat, args.output)
    maybe_plot_pca_value_features(model, states, z, args.output)

    # Summarize pass/fail for checkboxes
    print("\n=== CHECKBOX SUMMARY ===")
    print(f"[{'x' if in_range else ' '}] Value outputs in [-1,1] (approx)")
    print(
        f"[{'x' if pretanh_available and sat_share < 0.20 else ' '}] Pre-tanh saturation < 20% (if measurable)"
    )
    print(f"[{'x' if np.isfinite(bs) else ' '}] Brier computed")
    print(f"[{'x' if np.isfinite(ece) else ' '}] ECE computed")
    print("[x] Hist & scatter plots saved to:", args.output)

    # Clean up hook
    if hook_handle is not None:
        hook_handle.remove()


if __name__ == "__main__":
    main()
