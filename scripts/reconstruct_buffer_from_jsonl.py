#!/usr/bin/env python3
"""
Reconstruct a replay buffer from a JSONL self-play log file.

This script reads a JSONL file containing SampleV2 records and converts them
back into a ReplayBuffer pickle file that can be used for training.

Usage:
    python scripts/reconstruct_buffer_from_jsonl.py \
        --jsonl checkpoints/selfplay/selfplay_c1_cycle2.jsonl \
        --output checkpoints/buffers/replay_c1_cycle2.pkl \
        --max-size 30000
"""

import argparse
import json
from pathlib import Path

import torch

from train.replay_buffer import ReplayBuffer


def load_jsonl_samples(jsonl_path: str) -> list[dict]:
    """Load all training samples from JSONL, skipping game summaries."""
    samples = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                # Skip game_summary records
                if rec.get("type") == "game_summary":
                    continue
                # Only include records with state, pi_mcts, and v_scalar
                if "state" in rec and "pi_mcts" in rec and "v_scalar" in rec:
                    samples.append(rec)
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed line: {line[:50]}...")
                continue
    return samples


def convert_sample_to_buffer_entry(sample: dict) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Convert a JSON sample to a buffer entry tuple.

    Args:
        sample: Dictionary with 'state', 'pi_mcts', and 'v_scalar' keys

    Returns:
        Tuple of (state_tensor, pi_tensor, z_value)
    """
    # Convert state from nested list to tensor [3, 8, 8]
    state_list = sample["state"]
    state_tensor = torch.tensor(state_list, dtype=torch.float32)

    # Ensure correct shape: should be [3, 8, 8]
    if state_tensor.dim() == 3 and state_tensor.shape == (3, 8, 8):
        pass  # Correct shape
    elif state_tensor.dim() == 2:
        # If it's [8, 8], we need to add channel dimension - but this shouldn't happen
        raise ValueError(f"Unexpected state shape: {state_tensor.shape}")
    else:
        raise ValueError(f"Unexpected state shape: {state_tensor.shape}, expected [3, 8, 8]")

    # Convert pi_mcts from nested list to tensor [8, 8]
    pi_list = sample["pi_mcts"]
    pi_tensor = torch.tensor(pi_list, dtype=torch.float32)

    # Ensure correct shape: should be [8, 8]
    if pi_tensor.dim() == 2 and pi_tensor.shape == (8, 8):
        pass  # Correct shape
    elif pi_tensor.dim() == 1:
        # Reshape if it's flattened
        pi_tensor = pi_tensor.view(8, 8)
    else:
        raise ValueError(f"Unexpected pi_mcts shape: {pi_tensor.shape}, expected [8, 8]")

    # Extract v_scalar (should already be a float)
    z_value = float(sample["v_scalar"])

    # Validate z_value is in expected range
    if z_value not in [-1.0, 0.0, 1.0]:
        print(f"Warning: Unexpected z_value {z_value}, expected -1.0, 0.0, or 1.0")

    return (state_tensor, pi_tensor, z_value)


def reconstruct_buffer(
    jsonl_path: str,
    output_path: str,
    max_size: int,
    validate: bool = True,
) -> ReplayBuffer:
    """
    Reconstruct a ReplayBuffer from a JSONL file.

    Args:
        jsonl_path: Path to input JSONL file
        output_path: Path where buffer will be saved
        max_size: Maximum size of the replay buffer
        validate: Whether to validate the reconstructed buffer

    Returns:
        The reconstructed ReplayBuffer
    """
    print(f"Loading samples from {jsonl_path}...")
    samples = load_jsonl_samples(jsonl_path)
    print(f"Found {len(samples)} training samples")

    if len(samples) == 0:
        raise ValueError("No training samples found in JSONL file!")

    # Create buffer
    buffer = ReplayBuffer(max_size=max_size)

    # Convert samples to buffer entries
    print("Converting samples to buffer format...")
    buffer_entries: list[tuple[torch.Tensor, torch.Tensor, float]] = []

    for i, sample in enumerate(samples):
        try:
            entry = convert_sample_to_buffer_entry(sample)
            buffer_entries.append(entry)
        except Exception as e:
            print(f"Warning: Failed to convert sample {i}: {e}")
            continue

    print(f"Successfully converted {len(buffer_entries)} samples")

    # Add all entries to buffer at once (more efficient)
    buffer.buffer = buffer_entries

    # Trim if over capacity (FIFO)
    overflow = len(buffer.buffer) - max_size
    if overflow > 0:
        print(f"Trimming {overflow} samples (buffer capacity: {max_size})")
        buffer.buffer = buffer.buffer[overflow:]

    print(f"Buffer size: {len(buffer)} samples")

    # Validate buffer
    if validate:
        print("Validating buffer...")
        validate_buffer(buffer)

    # Save buffer
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    buffer.save(output_path)
    print(f"Buffer saved to {output_path}")

    return buffer


def validate_buffer(buffer: ReplayBuffer) -> None:
    """Validate the reconstructed buffer."""
    if len(buffer) == 0:
        raise ValueError("Buffer is empty!")

    # Sample a few entries to validate
    sample_size = min(10, len(buffer))
    states, policies, values = buffer.sample(sample_size)

    # Check shapes
    assert states.shape == (sample_size, 3, 8, 8), f"Unexpected states shape: {states.shape}"
    assert policies.shape == (sample_size, 8, 8), f"Unexpected policies shape: {policies.shape}"
    assert values.shape == (sample_size,), f"Unexpected values shape: {values.shape}"

    # Check value range
    unique_values = set(values.tolist())
    expected_values = {-1.0, 0.0, 1.0}
    unexpected = unique_values - expected_values
    if unexpected:
        print(f"Warning: Found unexpected value(s): {unexpected}")

    # Check policy sums (should sum to ~1.0 over legal moves)
    policy_sums = policies.sum(dim=(1, 2))
    # Policies might not sum to exactly 1.0 due to floating point, but should be close
    for i, s in enumerate(policy_sums):
        if not (0.9 <= s.item() <= 1.1):  # Allow some tolerance
            print(f"Warning: Policy {i} sum is {s.item()}, expected ~1.0")

    print("✓ Buffer validation passed")


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct a replay buffer from JSONL self-play data"
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        required=True,
        help="Path to input JSONL file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output pickle file",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=30000,
        help="Maximum size of the replay buffer (default: 30000)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip buffer validation",
    )

    args = parser.parse_args()

    # Check input file exists
    if not Path(args.jsonl).exists():
        raise FileNotFoundError(f"Input JSONL file not found: {args.jsonl}")

    # Reconstruct buffer
    buffer = reconstruct_buffer(
        jsonl_path=args.jsonl,
        output_path=args.output,
        max_size=args.max_size,
        validate=not args.no_validate,
    )

    print("\n" + "=" * 80)
    print("Reconstruction complete!")
    print(f"Buffer contains {len(buffer)} samples")
    print(f"Saved to: {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
