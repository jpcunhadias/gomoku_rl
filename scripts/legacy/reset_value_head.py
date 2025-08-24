# scripts/reset_value_head.py
import argparse
import os

import torch

from model.policy_value_net import PolicyValueNet


def reset_value_head(model):
    # Zero bias; Glorot/Xavier with small gain on final layer
    if hasattr(model, "value_fc"):
        torch.nn.init.xavier_uniform_(model.value_fc.weight, gain=0.8)
        torch.nn.init.zeros_(model.value_fc.bias)
    if hasattr(model, "value_conv"):
        # 1x1 conv before FC: standard kaiming (relu trunk feeds it)
        torch.nn.init.kaiming_normal_(model.value_conv.weight, nonlinearity="relu")
        if model.value_conv.bias is not None:
            torch.nn.init.zeros_(model.value_conv.bias)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_in", required=True)
    ap.add_argument("--ckpt_out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--board_size", type=int, default=8)
    args = ap.parse_args()

    model = PolicyValueNet.load_from_checkpoint(
        args.ckpt_in, board_size=args.board_size, device=args.device
    )
    reset_value_head(model)

    os.makedirs(os.path.dirname(args.ckpt_out) or ".", exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, args.ckpt_out)
    print(f"[OK] Saved model with reset value head to {args.ckpt_out}")
