import json

import numpy as np

path = "checkpoints/selfplay/selfplay_v2.jsonl"
ply0 = []
with open(path) as f:
    for line in f:
        rec = json.loads(line)
        if rec.get("type") == "game_summary":
            continue
        if rec.get("move_number") == 0:
            h = rec.get("entropy_pi_mcts")
            if h is not None:
                ply0.append(float(h))

ply0 = np.array(ply0)
print(
    f"n={len(ply0)}  mean={ply0.mean():.3f}  p10={np.percentile(ply0, 10):.3f}  p50={np.percentile(ply0, 50):.3f}"
)
print(f"share(H<0.10) = {(ply0 < 0.10).mean():.3%}")
print(f"share(H<0.25) = {(ply0 < 0.25).mean():.3%}")
