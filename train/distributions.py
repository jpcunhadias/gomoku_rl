import torch


def entropy_over_legal(pi: torch.Tensor, legal: torch.Tensor) -> float:
    # pi [8,8] probs on board; legal [8,8] bool
    p = pi[legal]
    p = p / (p.sum() + 1e-12)
    q = torch.clamp(p, 1e-12, 1.0)
    return float(-(q * q.log()).sum())


def kl_over_legal(net: torch.Tensor, mcts: torch.Tensor, legal: torch.Tensor) -> float:
    p = net[legal]
    p = p / (p.sum() + 1e-12)
    q = mcts[legal]
    q = q / (q.sum() + 1e-12)
    p = torch.clamp(p, 1e-12, 1.0)
    q = torch.clamp(q, 1e-12, 1.0)
    return float((p * (p.log() - q.log())).sum())
