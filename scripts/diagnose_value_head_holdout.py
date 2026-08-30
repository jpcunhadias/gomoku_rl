#!/usr/bin/env python3
"""
Value-head calibration diagnostic with a proper train/held-out split.

Every prior calibration check in this repo (debug/value_head_check.py) evaluates on samples
drawn from the same buffer the model trained on -- that conflates "memorized this buffer" with
"is actually calibrated." This script splits the Cycle 2 buffer 90/10, trains fresh copies of
the model (from the same Cycle 1 starting checkpoint) with different optimizer configs on the
90%, and reports Brier/ECE/pre-tanh-saturation on the untouched 10% for each — a genuine
generalization comparison, not a training-set readout.

Usage:
  uv run python scripts/diagnose_value_head_holdout.py \
      --buffer checkpoints/buffers/replay_c1_cycle2.pkl \
      --init_checkpoint checkpoints/models/c1_cycle1_last.pth
"""

import argparse
import copy
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from model.policy_value_net import PolicyValueNet
from train.replay_buffer import ReplayBuffer


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_buffer(buffer: ReplayBuffer, holdout_frac: float, seed: int):
    samples = list(buffer.buffer)
    rng = random.Random(seed)
    rng.shuffle(samples)
    n_holdout = int(len(samples) * holdout_frac)
    holdout = samples[:n_holdout]
    train = samples[n_holdout:]

    train_buffer = ReplayBuffer(max_size=len(train))
    train_buffer.buffer = train
    return train_buffer, holdout


def make_optimizer(model: PolicyValueNet, lr: float, value_weight_decay: float, use_adamw: bool):
    value_params = list(model.value_conv.parameters()) + list(model.value_fc.parameters())
    value_param_ids = {id(p) for p in value_params}
    policy_params = [p for p in model.parameters() if id(p) not in value_param_ids]

    groups = [
        {"params": policy_params, "lr": lr},
        {"params": value_params, "lr": lr * 0.3, "weight_decay": value_weight_decay},
    ]
    opt_cls = torch.optim.AdamW if use_adamw else torch.optim.Adam
    return opt_cls(groups)


def train_one_config(
    init_state_dict, device, train_buffer, epochs, steps_per_epoch, batch_size,
    lr, value_weight_decay, use_adamw, seed,
):
    set_seeds(seed)
    model = PolicyValueNet(board_size=8).to(device)
    model.load_state_dict(copy.deepcopy(init_state_dict))
    model.train()

    optimizer = make_optimizer(model, lr, value_weight_decay, use_adamw)
    policy_loss_fn = nn.KLDivLoss(reduction="batchmean")
    value_loss_fn = nn.SmoothL1Loss(beta=1.0)

    for epoch in range(1, epochs + 1):
        ep_p, ep_v = 0.0, 0.0
        for _ in range(steps_per_epoch):
            states, target_policies, target_values = train_buffer.sample(batch_size)
            states = states.to(device)
            target_policies = target_policies.to(torch.float32).to(device)
            target_policies = target_policies.view(-1, model.policy_fc.out_features)
            target_values = target_values.to(torch.float32).to(device)

            optimizer.zero_grad()
            logits, value_pred = model(states)
            log_probs = F.log_softmax(logits, dim=1)
            p_loss = policy_loss_fn(log_probs, target_policies)
            v_loss = value_loss_fn(value_pred.view(-1), target_values)
            loss = p_loss * 1.0 + v_loss * 0.5
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_p += p_loss.item()
            ep_v += v_loss.item()
        print(
            f"  epoch {epoch:2d}: policy {ep_p / steps_per_epoch:.4f}  "
            f"value {ep_v / steps_per_epoch:.4f}"
        )
    return model


def brier_score(v_hat: np.ndarray, z: np.ndarray) -> float:
    return float(np.mean((v_hat - z) ** 2))


def scalar_ece(v_hat: np.ndarray, z: np.ndarray, num_bins: int = 10) -> float:
    bins = np.linspace(-1.0, 1.0, num_bins + 1)
    indices = np.digitize(v_hat, bins) - 1
    total = 0.0
    for b in range(num_bins):
        mask = indices == b
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        gap = abs(float(v_hat[mask].mean()) - float(z[mask].mean()))
        total += gap * (cnt / len(v_hat))
    return total


def evaluate_on_holdout(model, device, holdout):
    model.eval()
    states, _, values = zip(*holdout)
    states = torch.stack(states).to(device)
    z = np.array(values, dtype=np.float32)

    pre = {}

    def hook(_module, _input, output):
        pre["v"] = output.detach().view(-1).cpu().numpy()

    handle = model.value_fc.register_forward_hook(hook)
    with torch.no_grad():
        _, value_pred = model(states)
    handle.remove()

    v_hat = value_pred.view(-1).detach().cpu().numpy()
    sat_share = float(np.mean(np.abs(pre["v"]) > 2.0))
    return {
        "n": len(holdout),
        "brier": brier_score(v_hat, z),
        "ece": scalar_ece(v_hat, z),
        "saturation": sat_share,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--buffer", required=True)
    ap.add_argument("--init_checkpoint", required=True)
    ap.add_argument("--holdout_frac", type=float, default=0.10)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--steps_per_epoch", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--value_weight_decay", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--split_seed", type=int, default=777)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")

    buffer = ReplayBuffer.load(args.buffer)
    train_buffer, holdout = split_buffer(buffer, args.holdout_frac, args.split_seed)
    print(f"[INFO] buffer={len(buffer)}  train={len(train_buffer)}  holdout={len(holdout)}")

    init_ckpt = torch.load(args.init_checkpoint, map_location=device)
    init_state_dict = init_ckpt["model_state_dict"]

    # Baseline: measure the starting checkpoint's own held-out calibration (before any
    # Cycle-2-style training at all), for reference.
    base_model = PolicyValueNet(board_size=8).to(device)
    base_model.load_state_dict(init_state_dict)
    base_stats = evaluate_on_holdout(base_model, device, holdout)
    print(f"\n[Cycle 1 checkpoint, no further training] {base_stats}")

    configs = [
        ("Adam (matches original Cycle 2 training)", False),
        ("AdamW (decoupled weight decay)", True),
    ]

    results = []
    for name, use_adamw in configs:
        print(f"\n=== {name} ===")
        model = train_one_config(
            init_state_dict, device, train_buffer,
            args.epochs, args.steps_per_epoch, args.batch_size,
            args.lr, args.value_weight_decay, use_adamw, args.seed,
        )
        stats = evaluate_on_holdout(model, device, holdout)
        print(f"  held-out: {stats}")
        results.append((name, stats))

    print("\n=== SUMMARY (held-out set, n={}) ===".format(len(holdout)))
    print(f"{'config':45s}  {'brier':>8s}  {'ece':>8s}  {'sat%':>6s}")
    print(f"{'Cycle 1 checkpoint (no training)':45s}  "
          f"{base_stats['brier']:8.4f}  {base_stats['ece']:8.4f}  "
          f"{base_stats['saturation']*100:5.1f}%")
    for name, stats in results:
        print(f"{name:45s}  {stats['brier']:8.4f}  {stats['ece']:8.4f}  "
              f"{stats['saturation']*100:5.1f}%")


if __name__ == "__main__":
    main()
