import torch

# 8 symmetries (match your augmentation order)
SYMS = [
    lambda t: t,
    lambda t: torch.rot90(t, 1, (1, 2)),
    lambda t: torch.rot90(t, 2, (1, 2)),
    lambda t: torch.rot90(t, 3, (1, 2)),
    lambda t: torch.flip(t, (1,)),  # vertical
    lambda t: torch.flip(t, (2,)),  # horizontal
    lambda t: torch.transpose(t, 1, 2),  # main diagonal
    lambda t: torch.flip(torch.transpose(t, 1, 2), (1,)),  # anti-diag
]


def canonicalize_state(state: torch.Tensor) -> torch.Tensor:
    """
    Ensure 'current player to move' is in plane 0 (your encoder already does this),
    so canonicalization is identity here. Keep as a hook in case encoding changes.
    """
    return state


def minhash_symmetries(state: torch.Tensor) -> int:
    """Return a hash that is invariant to board symmetries."""
    variants = []
    x = state.contiguous()
    for T in SYMS:
        v = T(x).contiguous().cpu().numpy().tobytes()
        variants.append(v)
    return hash(min(variants))
