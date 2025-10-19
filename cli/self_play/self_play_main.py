import argparse

from train.config import get_config
from train.self_play import run_selfplay_pipeline
from utils.seeding import set_global_seed


def main() -> None:
    """Run a single self-play pipeline for a specific cycle."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cycle", type=int, required=True, help="Cycle number for this self-play run"
    )
    args = parser.parse_args()

    print(f"[Self-Play] Starting run for cycle {args.cycle}")

    config = get_config()
    # Override the cycle from the config file with the one from the command line
    config.cycle = args.cycle

    set_global_seed(config.seed)

    run_selfplay_pipeline(
        config=config,
        load_checkpoint=True,  # Always load the latest policy
    )

    print(f"[Self-Play] Cycle {args.cycle} finished.")


if __name__ == "__main__":
    main()
