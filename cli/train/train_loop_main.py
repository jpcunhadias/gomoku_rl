import argparse
import os
import time
from types import SimpleNamespace

import torch

from model.policy_value_net import PolicyValueNet
from train.config import get_config
from train.replay_buffer import ReplayBuffer
from train.train_loop import run_training
from utils.paths import cycle_paths, save_config, save_json, save_meta, hash_config
from utils.seeding import set_global_seed


def main() -> None:
    # First, load the base config to dynamically create CLI arguments
    base_cfg = get_config()

    ap = argparse.ArgumentParser(
        description="Run the training loop with optional hyperparameter overrides."
    )
    ap.add_argument(
        "--cycle", type=int, required=True, help="Experiment cycle id (int)"
    )

    # Dynamically add arguments for each parameter in the base config
    for key, value in vars(base_cfg).items():
        arg_type = type(value)
        if arg_type == bool:
            # Handle boolean flags properly
            ap.add_argument(
                f"--{key}",
                action=argparse.BooleanOptionalAction,
                default=None, # Default to None to distinguish from user-provided False
                help=f"Override {key} (default: {value})",
            )
        elif isinstance(value, (dict, SimpleNamespace, list)):
            pass
        else:
            ap.add_argument(
                f"--{key}",
                type=arg_type,
                default=None,
                help=f"Override {key} (default: {value})",
            )

    args = ap.parse_args()

    # Start with the base config
    cfg = base_cfg
    overrides = {}

    # Apply overrides from CLI arguments
    for key in vars(base_cfg).keys():
        if hasattr(args, key):
            cli_value = getattr(args, key)
            if cli_value is not None:
                original_value = getattr(cfg, key)
                if cli_value != original_value:
                    print(f"[Config Override] {key}: {original_value} -> {cli_value}")
                    setattr(cfg, key, cli_value)
                    overrides[key] = {"from": original_value, "to": cli_value}

    # Set the global seed for reproducibility
    set_global_seed(cfg.seed)

    # --- Hashing and metadata saving ---
    config_hash = hash_config(cfg)
    cfg.config_hash = config_hash
    print(f"[Config] Final config hash: {config_hash}")

    paths = cycle_paths(args.cycle)

    print("Starting AlphaZero training loop")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- persist config & meta stub up-front
    save_config(cfg, paths["config"])
    meta = save_meta(
        cycle=args.cycle,
        seed=getattr(cfg, "seed", 42),
        notes="train loop start",
        extra={"config_hash": config_hash, "overrides": overrides},
    )
    save_json(meta, paths["meta"])

    # === Initialize model & optimizer ===
    model = PolicyValueNet(board_size=8)
    model._init_weights()
    model.to(device)

    value_params = list(model.value_conv.parameters()) + list(
        model.value_fc.parameters()
    )
    value_param_ids = {id(p) for p in value_params}
    policy_params = [p for p in model.parameters() if id(p) not in value_param_ids]

    optimizer = torch.optim.Adam(
        [
            {"params": policy_params, "lr": cfg.learning_rate},
            {
                "params": value_params,
                "lr": cfg.learning_rate * 0.3,
                "weight_decay": 2e-4,
            },
        ]
    )

    # === Load buffer (cycle-aware) ===
    buffer_path = str(paths["buffer"])
    if os.path.exists(buffer_path):
        replay_buffer = ReplayBuffer.load(buffer_path)
        print(f"Loaded buffer with {len(replay_buffer)} samples from {buffer_path}")
    else:
        raise FileNotFoundError(f"No replay buffer at {buffer_path}")

    # choose resume checkpoint
    resume_path = None
    if os.path.exists(paths["model_last"]):
        resume_path = paths["model_last"]
    elif os.path.exists(paths["model_best"]):
        resume_path = paths["model_best"]

    best_value_loss = float("inf")
    if resume_path:
        print(f"Loading checkpoint from: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_value_loss = checkpoint.get("best_value_loss", float("inf"))

    # === Train ===
    t0 = time.time()
    best_epoch, best_val = run_training(
        model=model,
        optimizer=optimizer,
        buffer=replay_buffer,
        config=cfg,
        config_hash=config_hash,  # Pass hash to training loop
        best_value_loss=best_value_loss,
        debug=True,
        save_paths=paths,
    )
    t1 = time.time()

    # --- tiny summary + meta update
    tiny_summary = {
        "cycle": args.cycle,
        "best_epoch": best_epoch,
        "best_value_loss": best_val,
        "elapsed_sec": t1 - t0,
        "model_best": str(paths["model_best"]),
        "model_last": str(paths["model_last"]),
        "buffer": str(paths["buffer"]),
        "config_hash": config_hash,
    }
    save_json(tiny_summary, paths["diag_smoke"])  # reuse diag_smoke for the summary

    meta_end = save_meta(
        cycle=args.cycle,
        seed=getattr(cfg, "seed", 42),
        notes="train loop end",
        extra={
            "elapsed_sec": t1 - t0,
            "config_hash": config_hash,
            "overrides": overrides,
        },
    )
    save_json(meta_end, paths["meta"])


if __name__ == "__main__":
    main()
