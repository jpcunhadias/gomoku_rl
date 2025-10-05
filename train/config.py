import os
from importlib import import_module
from types import SimpleNamespace


def _inline_default_config() -> SimpleNamespace:
    from configs.phaseC_c1 import get_config as _c1

    return _c1()


def get_config() -> SimpleNamespace:
    profile = os.getenv("CFG_PROFILE", "").strip()
    if profile:
        mod = import_module(f"configs.{profile}")
        return mod.get_config()
    return _inline_default_config()
