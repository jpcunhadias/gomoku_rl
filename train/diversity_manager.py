import random
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from train.bucketer import BucketKey, bucket_key

# Type alias: a training tuple and its metadata (move_number, z)
Sample = Tuple[torch.Tensor, torch.Tensor, float]  # (state, pi, z)
Meta = Tuple[int, int, float]  # (local_idx, move_no, z_scalar)


class DiversityManager:
    """
    Online, light-touch admission controller.
    Keeps per-bucket counts and only admits samples while buckets are under target.
    When full, does simple reservoir-style replacement with small probability.
    """

    def __init__(
        self, target_quota: Dict[BucketKey, int], rng: Optional[random.Random] = None
    ):
        self.target_quota = target_quota
        self.counts: Dict[BucketKey, int] = defaultdict(int)
        self.rng = rng or random.Random(0)

    @staticmethod
    def default_targets(window_size: int = 30000) -> Dict[BucketKey, int]:
        """
        Target absolute counts per bucket for a ~30k window.
        Heuristic: balance outcomes roughly, bias midgame slightly.
        """
        fracs = {
            # WIN buckets
            ("win", "early"): 0.1093333333,  # 0.08 * 1.3666667
            ("win", "mid"): 0.1913333333,  # 0.14 * 1.3666667
            ("win", "late"): 0.1093333333,  # 0.08 * 1.3666667
            # LOSS buckets
            ("loss", "early"): 0.1093333333,
            ("loss", "mid"): 0.1913333333,
            ("loss", "late"): 0.1093333333,
            # DRAW buckets
            ("draw", "early"): 0.0,
            ("draw", "mid"): 0.0,
            ("draw", "late"): 0.0,
        }
        targets = {}
        for (o, p), f in fracs.items():
            targets[BucketKey(o, p)] = int(window_size * f)
        return targets

    def admit_batch(
        self,
        samples: Iterable[Sample],
        metas: Iterable[Meta],
    ) -> Tuple[List[Sample], List[Meta]]:
        """
        Admit samples by bucket (move_no, z_scalar).
        Meta is (local_idx, move_no, z_scalar) so the caller can mark exactly which
        per-move logs were accepted.
        """
        accepted_samples: List[Sample] = []
        accepted_metas: List[Meta] = []

        for (s, m, z), (i, move_no, z_meta) in zip(samples, metas):
            key = bucket_key(int(move_no), float(z_meta))
            tgt = self.target_quota.get(key, 0)
            cur = self.counts[key]

            if cur < tgt:
                self.counts[key] += 1
                accepted_samples.append((s, m, z))
                accepted_metas.append((i, move_no, z_meta))
            else:
                denom = max(10 * tgt, 1)
                p = 1.0 / denom
                if self.rng.random() < p:
                    self.counts[key] += 1
                    accepted_samples.append((s, m, z))
                    accepted_metas.append((i, move_no, z_meta))

        return accepted_samples, accepted_metas

    def snapshot_counts(self) -> Dict[BucketKey, int]:
        return dict(self.counts)
