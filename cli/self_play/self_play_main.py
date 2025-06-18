import time

from train.config import get_config
from train.self_play import run_selfplay_pipeline

config = get_config()


def main() -> None:
    """Continuously run self-play and update the replay buffer."""
    print("Self-play worker started.")

    while True:
        model, buffer = run_selfplay_pipeline(
            config=config,
            load_checkpoint=True,  # Always load the latest policy
            buffer_save_path="checkpoints/replay_buffer.pkl",
        )

        print("[Self-Play] Batch finished, sleeping for a while...")
        time.sleep(30)  # Sleep 30s to avoid hammering disk / CPU too fast


if __name__ == "__main__":
    main()
