import argparse
from types import SimpleNamespace

from train.config import get_config


def get_config_and_override(description: str) -> SimpleNamespace:
    base_cfg = get_config()

    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--cycle", type=int, required=True, help="Experiment cycle id (int)")
    ap.add_argument(
        "--config",
        type=str,
        default=None,
        help="Name of the config file to use (e.g., 'phaseC_c1')",
    )

    # Dynamically add arguments for each parameter in the base config
    for key, value in vars(base_cfg).items():
        if key in ["cycle", "config"]:
            continue
        arg_type = type(value)
        if arg_type is bool:
            ap.add_argument(
                f"--{key}",
                action=argparse.BooleanOptionalAction,
                default=None,
                help=f"Override {key} (default: {value})",
            )
        elif isinstance(value, dict):
            for subkey in value.keys():
                nested_arg = f"--{key}.{subkey}"
                ap.add_argument(
                    nested_arg,
                    type=type(value[subkey]),
                    default=None,
                    dest=f"{key}_{subkey}",
                    help=f"Override {key}[{subkey}] (default: {value[subkey]})",
                )
        elif isinstance(value, (SimpleNamespace, list)):
            pass
        else:
            ap.add_argument(
                f"--{key}",
                type=arg_type,
                default=None,
                help=f"Override {key} (default: {value})",
            )

    args, _ = ap.parse_known_args()

    # Load the specified config if provided, otherwise use the default
    cfg = get_config(args.config) if args.config else base_cfg
    overrides = {}

    # Apply overrides from CLI arguments
    for key in vars(cfg).keys():
        if hasattr(args, key):
            cli_value = getattr(args, key)
            if cli_value is not None:
                original_value = getattr(cfg, key)
                if cli_value != original_value:
                    print(f"[Config Override] {key}: {original_value} -> {cli_value}")
                    setattr(cfg, key, cli_value)
                    overrides[key] = {"from": original_value, "to": cli_value}

    # Handle nested dict overrides
    for key, value in vars(cfg).items():
        if isinstance(value, dict):
            nested_dict = dict(value)
            modified = False
            for subkey in value.keys():
                nested_arg_name = f"{key}_{subkey}"
                if hasattr(args, nested_arg_name):
                    cli_value = getattr(args, nested_arg_name)
                    if cli_value is not None:
                        original_value = nested_dict[subkey]
                        if cli_value != original_value:
                            print(
                                f"[Config Override] {key}[{subkey}]: {original_value} -> {cli_value}"
                            )
                            nested_dict[subkey] = cli_value
                            modified = True
                            if key not in overrides:
                                overrides[key] = {"from": dict(value), "to": {}}
                            overrides[key]["to"][subkey] = cli_value
            if modified:
                setattr(cfg, key, nested_dict)

    cfg.cycle = args.cycle
    return cfg, overrides, args
