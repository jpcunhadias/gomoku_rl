import json

import numpy as np

path = "checkpoints/selfplay_v2.jsonl"
n, bad_sum, bad_illegal = 0, 0, 0
with open(path) as f:
    for line in f:
        n += 1
        rec = json.loads(line)
        pi = np.array(rec["pi_mcts"], dtype=np.float32)
        legal = np.array(rec["legal_mask"], dtype=bool)
        if np.any(pi[~legal] > 1e-8):
            bad_illegal += 1
        s = pi[legal].sum() if legal.any() else 0.0
        if legal.any() and not np.isclose(s, 1.0, atol=1e-6):
            bad_sum += 1
print(f"records={n}  illegal>0={bad_illegal}  sum(legal)!=1={bad_sum}")
assert bad_illegal == 0, "Some pi values are non-zero on illegal moves."
assert bad_sum == 0, "Some pi values do not sum to 1 over legal moves."
print("Sanity checks passed.")
