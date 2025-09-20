import random
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from train.bucketer import BucketKey, bucket_key

# Type alias: a training tuple and its metadata (move_number, z)
Sample = Tuple  # (state_tensor, pi_tensor, z)
Meta = Tuple[int, float]  # (move_number, z)


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
        self, samples: Iterable[Sample], metas: Iterable[Meta]
    ) -> List[Sample]:
        """
        Decide which samples to admit.
        If a bucket is below target → admit.
        If at/above target → admit with small prob (reservoir trick).
        """
        accepted: List[Sample] = []
        for (s, m, z), (move_no, z_meta) in zip(samples, metas):
            assert float(z) == float(z_meta), "z mismatch between sample and meta"
            key = bucket_key(move_no, float(z))
            tgt = self.target_quota.get(key, 0)
            cur = self.counts[key]

            if cur < tgt:
                self.counts[key] += 1
                accepted.append((s, m, z))
            else:
                # small admission probability keeps a trickle of freshness
                # scale ~ 1 / (10 * bucket_target) to be conservative
                p = 1.0 / max(10 * (tgt if tgt > 0 else 1), 1)
                if self.rng.random() < p:
                    # accept and conceptually "replace" one from the bucket
                    self.counts[key] += 1
                    accepted.append((s, m, z))
                # else reject
        return accepted

    def snapshot_counts(self) -> Dict[BucketKey, int]:
        return dict(self.counts)
