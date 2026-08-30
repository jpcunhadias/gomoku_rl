import json
import subprocess
from datetime import datetime
from pathlib import Path
import hashlib

BASE = Path("checkpoints")


def hash_config(config_obj) -> str:
    """Create a SHA1 hash of a configuration object for traceability."""
    # Convert SimpleNamespace to dict if necessary
    data = vars(config_obj) if hasattr(config_obj, '__dict__') else config_obj
    
    # Serialize to a canonical JSON string (sorted keys, no whitespace)
    canonical_json = json.dumps(data, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    
    # Return the SHA1 hash
    return hashlib.sha1(canonical_json).hexdigest()


def short_sha():
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        return "nogit"


def cycle_paths(cycle: int):
    p = {
        "model_best": BASE / "models" / f"c1_cycle{cycle}_best.pth",
        "model_last": BASE / "models" / f"c1_cycle{cycle}_last.pth",
        "buffer": BASE / "buffers" / f"replay_c1_cycle{cycle}.pkl",
        "config": BASE / "configs" / f"c1_cycle{cycle}.json",
        "meta": BASE / "meta" / f"c1_cycle{cycle}_meta.json",
        "sp_log": BASE / "selfplay" / f"selfplay_c1_cycle{cycle}.jsonl",
        "sp_summary": BASE / "selfplay" / f"c1_cycle{cycle}_summary.json",
        "diag_dir": BASE / "diagnostics" / f"c1_cycle{cycle}_plots",
        "diag_policy": BASE / "diagnostics" / f"c1_cycle{cycle}_policy_head.json",
        "diag_value": BASE / "diagnostics" / f"c1_cycle{cycle}_value_head.json",
        "diag_smoke": BASE / "diagnostics" / f"c1_cycle{cycle}_train_smoke.json",
        "arena_json": BASE / "arena" / f"arena_c1_cycle{cycle}_vs_phaseB.json",
        "arena_log": BASE / "arena" / "arena_log.csv",
    }
    # ensure directories
    for k, v in p.items():
        (v.parent).mkdir(parents=True, exist_ok=True)
    return p


def save_config(cfg, path):
    with open(path, "w") as f:
        json.dump(vars(cfg) if hasattr(cfg, "__dict__") else cfg, f, indent=2)


def save_meta(cycle, seed, notes="", extra=None):
    meta = {
        "cycle": cycle,
        "seed": seed,
        "git_commit": short_sha(),
        "notes": notes,
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    if extra:
        meta.update(extra)
    return meta


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
