import os
from importlib import import_module
from types import SimpleNamespace


def _inline_default_config() -> SimpleNamespace:
    from configs.phaseC_c1 import get_config as _c1

    return _c1()


def get_config(name: str = None) -> SimpleNamespace:
    config_name = name or os.getenv("CFG_PROFILE", "").strip()
    if config_name:
        try:
            mod = import_module(f"configs.{config_name}")
            return mod.get_config()
        except ImportError:
            print(f"Warning: Config profile '{config_name}' not found. Falling back to default.")
    return _inline_default_config()
