from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SampleV2:
    # core training tensors
    state: torch.Tensor  # [3,8,8], float32
    pi_mcts: torch.Tensor  # [8,8], float32, zeros on illegal, sum over legal=1
    v_scalar: float  # -1.0, 0.0, +1.0

    # context
    legal_mask: torch.Tensor  # [8,8], bool
    move_number: int
    sims: int
    tau: float
    c_puct: float
    dirichlet_alpha: float
    dirichlet_eps: float

    # diagnostics (optional now, useful soon)
    entropy_pi_mcts: float  # over legal moves
    entropy_pi_net: float  # over legal moves
    kl_net_mcts: float  # KL(net || mcts) over legal

    # symmetry & canonicalization (filled later)
    symmetry_id: int  # 0..7 (as recorded)
    canon_hash: Optional[int]  # None for now

    # Indicates the version of the SampleV2 schema. Increment this value whenever the structure of the dataclass changes in a way that affects serialization, deserialization, or compatibility with stored data.
    schema_version: int = 2
