import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from model.policy_value_net import PolicyValueNet
from train.replay_buffer import ReplayBuffer

# --- Repro ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256
STEPS = 800  # 600–800 steps works well
BASE_LR = 1e-3
VALUE_LR_SCALE = 0.05  # slower value head
VALUE_WD = 2e-4
GRAD_CLIP = 1.0

# Regularizers
LAMBDA_CENTER = 5e-2  # penalize mean(v̂^2) to keep near 0
LAMBDA_HINGE = 1e-2  # penalize |pre_tanh| > 2 via hinge
PRE_TANH_MARGIN = 2.0

# Target scaling curriculum: 0.7 → 1.0 over the run
START_SCALE = 0.7
END_SCALE = 1.0

CKPT_IN = "checkpoints/policy_value_net_reset_value.pth"
BUFFER_P = "checkpoints/replay_buffer.pkl"
CKPT_OUT = "checkpoints/policy_value_net_after_micro_v3.pth"


def attach_pretanh_hook(model):
    store = {"pre": None}
    layer = getattr(model, "value_fc", None)
    if layer is None:
        return None, lambda: None

    def hook_fn(_m, _i, o):
        store["pre"] = o.detach()

    h = layer.register_forward_hook(hook_fn)
    return h, lambda: store["pre"]


def make_optimizer(model):
    value_params, other_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("value_conv") or n.startswith("value_fc"):
            value_params.append(p)
        else:
            other_params.append(p)
    return optim.AdamW(
        [
            {"params": other_params, "lr": BASE_LR, "weight_decay": 0.0},
            {
                "params": value_params,
                "lr": BASE_LR * VALUE_LR_SCALE,
                "weight_decay": VALUE_WD,
            },
        ]
    )


def kl_divergence_logits(logits, target_probs):
    log_probs = F.log_softmax(logits, dim=1)
    return F.kl_div(log_probs, target_probs, reduction="batchmean")


def smooth_l1_loss_weighted(pred, target, weights):
    # per-sample SmoothL1 with weights
    loss = F.smooth_l1_loss(pred.view(-1), target.view(-1), reduction="none")
    loss = (loss * weights.view(-1)).mean()
    return loss


def fetch_batch(buffer: ReplayBuffer, device):
    states, policies, values = buffer.sample(BATCH_SIZE)
    values = values.to(torch.float32)
    uniq = set(values.view(-1).tolist())
    if uniq.issubset({0.0, 1.0, 2.0}):
        values = values - 1.0  # legacy map
    policies = policies.view(policies.size(0), -1).to(torch.float32)
    return states.to(device), policies.to(device), values.to(device)


def linear_scale(step, steps, start, end):
    t = min(max(step / steps, 0.0), 1.0)
    return start + t * (end - start)


def main():
    model = PolicyValueNet.load_from_checkpoint(
        CKPT_IN, board_size=8, device=DEVICE
    ).train()
    buffer = ReplayBuffer.load(BUFFER_P)
    opt = make_optimizer(model)

    # policy:value ≈ 4:1
    Wp, Wv = 1.0, 0.25

    hook, get_pre = attach_pretanh_hook(model)

    for step in range(1, STEPS + 1):
        states, target_pi, target_v = fetch_batch(buffer, DEVICE)

        # curriculum: ramp targets toward full magnitude
        scale = linear_scale(step, STEPS, START_SCALE, END_SCALE)
        target_v_scaled = scale * target_v

        # draw-upweighting: emphasize samples near 0
        # weight = 2.0 for |target|<0.1, else 1.0
        draw_w = (target_v.abs() < 0.1).to(torch.float32) * 1.0 + 1.0

        opt.zero_grad(set_to_none=True)
        policy_logits, value_pred = model(states)

        loss_policy = kl_divergence_logits(policy_logits, target_pi)
        loss_value = smooth_l1_loss_weighted(value_pred, target_v_scaled, draw_w)

        # center penalty on outputs
        center_pen = (value_pred.view(-1) ** 2).mean()

        # hinge penalty on pre-tanh > margin
        pre = get_pre()
        if pre is not None:
            hinge = torch.relu(pre.abs() - PRE_TANH_MARGIN).view(-1)
            hinge_pen = hinge.mean()
        else:
            hinge_pen = torch.tensor(0.0, device=DEVICE)

        loss = (
            Wp * loss_policy
            + Wv * loss_value
            + LAMBDA_CENTER * center_pen
            + LAMBDA_HINGE * hinge_pen
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()

        if step % 20 == 0:
            with torch.no_grad():
                vmin = float(value_pred.min().item())
                vmax = float(value_pred.max().item())
                mean_abs = float(value_pred.abs().mean().item())
                sat = (
                    float((pre.abs() > PRE_TANH_MARGIN).float().mean().item())
                    if pre is not None
                    else float("nan")
                )
                draws = int((target_v.abs() < 0.1).sum().item())
            print(
                f"step {step:4d} | loss {loss.item():.4f} | P {loss_policy.item():.4f} | V {loss_value.item():.4f} "
                f"| v̂[{vmin:.3f},{vmax:.3f}] | |v̂|̄ {mean_abs:.3f} | sat>|{PRE_TANH_MARGIN}| {sat:.3f} | draws {draws} | scale {scale:.3f}"
            )

    os.makedirs(os.path.dirname(CKPT_OUT) or ".", exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, CKPT_OUT)
    if hook is not None:
        hook.remove()
    print(f"[OK] micro-train v3 finished → {CKPT_OUT}")


if __name__ == "__main__":
    main()
