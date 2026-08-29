import random

import torch

from train.bucketer import BucketKey
from train.diversity_manager import DiversityManager


def _sample(z: float):
    state = torch.zeros(1)
    pi = torch.zeros(1)
    return (state, pi, z)


def test_zero_quota_bucket_is_never_admitted():
    """A bucket with target_quota == 0 (e.g. draws) must be fully excluded,
    not admitted at ~100% as the old `1.0 / max(10 * tgt, 1)` formula did."""
    targets = {BucketKey("draw", "mid"): 0}
    manager = DiversityManager(targets, rng=random.Random(0))

    samples = [_sample(0.0) for _ in range(500)]
    metas = [(i, 20, 0.0) for i in range(500)]  # move 20 -> "mid" phase, z=0 -> "draw"

    accepted_samples, accepted_metas = manager.admit_batch(samples, metas)

    assert accepted_samples == []
    assert accepted_metas == []
    assert manager.snapshot_counts().get(BucketKey("draw", "mid"), 0) == 0


def test_nonzero_quota_bucket_admits_up_to_target_then_trickles():
    targets = {BucketKey("win", "mid"): 2}
    manager = DiversityManager(targets, rng=random.Random(0))

    samples = [_sample(1.0) for _ in range(10)]
    metas = [(i, 20, 1.0) for i in range(10)]

    accepted_samples, _ = manager.admit_batch(samples, metas)

    # exactly fills to quota, plus possibly a few reservoir trickle admits
    assert len(accepted_samples) >= 2
    assert manager.counts[BucketKey("win", "mid")] == len(accepted_samples)
