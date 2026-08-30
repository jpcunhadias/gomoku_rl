import json
import os
from typing import Any

import numpy as np
import torch


class SampleLogger:
    def __init__(self, out_path: str = "checkpoints/selfplay_v2.jsonl"):
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        self.out_path = out_path

    @staticmethod
    def _to_serializable(x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy().tolist()
        if isinstance(x, (np.ndarray,)):
            return x.tolist()
        return x

    def write(self, rec: dict[str, Any]):
        with open(self.out_path, "a") as f:
            json.dump({k: self._to_serializable(v) for k, v in rec.items()}, f)
            f.write("\n")
