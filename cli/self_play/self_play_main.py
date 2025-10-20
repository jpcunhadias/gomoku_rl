import argparse

from train.config import get_config
from train.self_play import run_selfplay_pipeline
from utils.seeding import set_global_seed


def _parse_and_update_config(config, unknown_args):
    i = 0
    while i < len(unknown_args):
        arg = unknown_args[i]
        if arg.startswith("--"):
            arg_name = arg[2:]
            if "." in arg_name:
                key, sub_key = arg_name.split(".", 1)
                if not hasattr(config, key) or not isinstance(getattr(config, key), dict):
                    setattr(config, key, {})
                if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith("--"):
                    value_str = unknown_args[i + 1]
                    try:
                        value = int(value_str)
                    except ValueError:
                        try:
                            value = float(value_str)
                        except ValueError:
                            value = value_str
                    
                    # The sub_key from split is a string, but tau_early_plies has int keys
                    try:
                        int_sub_key = int(sub_key)
                        getattr(config, key)[int_sub_key] = value
                    except ValueError:
                        getattr(config, key)[sub_key] = value
                    i += 2
                else:
                    i += 1
            else:
                if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith("--"):
                    value_str = unknown_args[i + 1]
                    if value_str.lower() == "true":
                        value = True
                    elif value_str.lower() == "false":
                        value = False
                    else:
                        try:
                            value = int(value_str)
                        except ValueError:
                            try:
                                value = float(value_str)
                            except ValueError:
                                value = value_str
                    setattr(config, arg_name, value)
                    i += 2
                else:
                    setattr(config, arg_name, True)
                    i += 1
        else:
            i += 1


def main() -> None:
    """Run a single self-play pipeline for a specific cycle."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cycle", type=int, required=True, help="Cycle number for this self-play run"
    )
    args, unknown_args = parser.parse_known_args()

    print(f"[Self-Play] Starting run for cycle {args.cycle}")

    config = get_config()
    config.cycle = args.cycle

    if unknown_args:
        _parse_and_update_config(config, unknown_args)

    set_global_seed(config.seed)

    print(
        "[SP config]",
        "uniform_root_p=",
        config.sp_uniform_root_p,
        "block_opening_repeats=",
        config.sp_block_opening_repeats,
        "opening_memory_size=",
        config.opening_memory_size,
        "restart_cap_fraction=",
        config.restart_cap_fraction,
        "dirichlet_epsilon_root=",
        config.dirichlet_epsilon_root,
        "tau_early_plies=",
        getattr(config, "tau_early_plies", None),
    )

    run_selfplay_pipeline(
        config=config,
        load_checkpoint=True,  # Always load the latest policy
    )

    print(f"[Self-Play] Cycle {args.cycle} finished.")


if __name__ == "__main__":
    main()