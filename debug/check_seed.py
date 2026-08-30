import argparse

import torch

from train.replay_buffer import ReplayBuffer
from utils.paths import cycle_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cycle", type=int, required=True, help="Cycle number of the buffer to check"
    )
    args = parser.parse_args()

    paths = cycle_paths(args.cycle)
    buffer_path = paths["buffer"]

    print(f"--- Checking Seeded Buffer for Cycle {args.cycle} ---")
    print(f"Loading buffer from: {buffer_path}")

    try:
        buffer = ReplayBuffer.load(str(buffer_path))
    except FileNotFoundError:
        print(f"ERROR: Buffer file not found at {buffer_path}")
        return

    # 1. Check buffer length
    # Note: This length is *after* the self-play run. It should be > seed_count.
    print("\n[Check 1] Buffer Length")
    print(f"  - Current buffer length: {len(buffer)}")

    # 2. Check target values
    print("\n[Check 2] Target Values")
    states, policies, values = buffer.sample(min(512, len(buffer)))
    unique_values = torch.unique(values).tolist()
    print(f"  - Unique target values in sample: {unique_values}")
    if not set(unique_values).issubset({-1.0, 0.0, 1.0}):
        print("  - WARNING: Target values are outside the expected {-1, 0, 1} set.")
    else:
        print("  - OK: Target values are in the expected {-1, 0, 1} set.")


if __name__ == "__main__":
    main()
