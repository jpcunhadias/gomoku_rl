import json

import torch

from train.canonicalize import SYMS, minhash_symmetries

path = "checkpoints/selfplay_v2.jsonl"
with open(path) as f:
    line = next(iter(f))
rec = json.loads(line)
s = torch.tensor(rec["state"], dtype=torch.float32)

h0 = minhash_symmetries(s)
mismatch = 0
for i, T in enumerate(SYMS):
    h = minhash_symmetries(T(s))
    if h != h0:
        print(f"Sym {i} mismatch: {h} != {h0}")
        mismatch += 1
print("OK" if mismatch == 0 else f"{mismatch} mismatches")
