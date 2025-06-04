import pickle
from collections import Counter


def main() -> None:
    """Load a replay buffer and print the distribution of ``z`` values."""

    with open("checkpoints/replay_buffer.pkl", "rb") as f:
        buffer = pickle.load(f)

    # Handle both ReplayBuffer object or list directly
    try:
        data = buffer.buffer  # assume it's a ReplayBuffer object
    except AttributeError:
        data = buffer  # assume it's a raw list of (state, pi, z)

    z_values = [z for _, _, z in data]
    print("z distribution:", Counter(z_values))


if __name__ == "__main__":
    main()
