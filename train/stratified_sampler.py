import json
import math
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from train.bucketer import BucketKey, bucket_key

TARGET_MIX = {
    # WIN (sum 0.5)
    BucketKey("win", "early"): 0.125,  # 0.10 / 0.80
    BucketKey("win", "mid"): 0.250,  # 0.20 / 0.80
    BucketKey("win", "late"): 0.125,  # 0.10 / 0.80
    # LOSS (sum 0.5)
    BucketKey("loss", "early"): 0.125,
    BucketKey("loss", "mid"): 0.250,
    BucketKey("loss", "late"): 0.125,
    # DRAW
    BucketKey("draw", "early"): 0.0,
    BucketKey("draw", "mid"): 0.0,
    BucketKey("draw", "late"): 0.0,
}

assert abs(sum(TARGET_MIX.values()) - 1.0) < 1e-9


class StratifiedBatchSampler:
    """
    Builds per-bucket index lists aligned to the current ReplayBuffer
    by reading the JSONL sidecar and keeping only admitted==1 records.
    Assumes JSONL append order for admitted matches buffer append order.
    """

    def __init__(
        self,
        sidecar_jsonl: str,
        buffer_len_fn,
        target_mix: Dict[BucketKey, float],
        rng: Optional[random.Random] = None,
        refresh_every: int = 1000,
    ) -> None:
        self.sidecar_jsonl = sidecar_jsonl
        self.buffer_len_fn = buffer_len_fn
        self.refresh_every = refresh_every
        self.rng = rng or random.Random(0)

        s = sum(target_mix.values())
        self.target_mix = {k: v / s for k, v in target_mix.items()}  # normalize

        self._bucket_indices: Dict[BucketKey, List[int]] = defaultdict(list)
        self._calls = 0
        self._last_buffer_len = -1
        self._idx_to_bucket: Dict[int, BucketKey] = {}
        self._epoch_counts: Counter = Counter()

    def _load_tail_aligned(self) -> List[Tuple[int, dict]]:
        buf_len = self.buffer_len_fn()
        self._last_buffer_len = buf_len

        # Read admitted records
        admitted: List[dict] = []
        with open(self.sidecar_jsonl, "r") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if int(rec.get("admitted", 0)) == 1:
                    admitted.append(rec)

        tail = admitted[-buf_len:] if buf_len <= len(admitted) else admitted
        return list(enumerate(tail))  # (buffer_idx, rec)

    def refresh(self) -> None:
        self._bucket_indices.clear()
        self._idx_to_bucket.clear()
        for idx, rec in self._load_tail_aligned():
            z = float(rec.get("v_scalar", 0.0))
            mn = int(rec.get("move_number", 0))
            key = bucket_key(mn, z)
            self._bucket_indices[key].append(idx)
            self._idx_to_bucket[idx] = key

    def begin_epoch(self) -> None:
        self._epoch_counts.clear()

    def end_epoch_report(self) -> Dict[BucketKey, int]:
        out = dict(self._epoch_counts)
        self._epoch_counts = Counter()
        return out

    def _ensure_fresh(self) -> None:
        cur_len = self.buffer_len_fn()
        if self._calls % self.refresh_every == 0 or cur_len != self._last_buffer_len:
            self.refresh()

    def sample_indices(self, batch_size: int) -> List[int]:
        self._calls += 1
        self._ensure_fresh()

        # desired counts
        desired: Dict[BucketKey, int] = {}
        rem = batch_size
        for k, frac in self.target_mix.items():
            n = int(math.floor(frac * batch_size))
            desired[k] = n
            rem -= n
        # distribute remainder by availability
        if rem > 0:
            by_avail = sorted(
                self._bucket_indices.items(), key=lambda kv: len(kv[1]), reverse=True
            )
            i = 0
            while rem > 0 and by_avail:
                k = by_avail[i % len(by_avail)][0]
                desired[k] = desired.get(k, 0) + 1
                rem -= 1
                i += 1

        picked: List[int] = []
        used = set()  # avoid dup within batch

        def _record(idx: int) -> None:
            bk = self._idx_to_bucket.get(idx)
            if bk is not None:
                self._epoch_counts[bk] += 1

        def take_from(key: BucketKey, need: int) -> int:
            got = 0
            pool = self._bucket_indices.get(key, [])
            # with replacement across steps, but no dup in a batch
            tries = 0
            while got < need and pool and tries < max(1, need * 5):
                idx = self.rng.choice(pool)
                tries += 1
                if idx not in used:
                    used.add(idx)
                    picked.append(idx)
                    got += 1
                    _record(idx)
            return got

        def fallback_fill(k: BucketKey, need: int) -> None:
            # neighbor phases (same outcome)
            order = {"early": 0, "mid": 1, "late": 2}
            phases = ["early", "mid", "late"]
            p0 = order[k.phase]
            for p in [p0 - 1, p0 + 1]:
                if 0 <= p < 3 and need > 0:
                    got = take_from(BucketKey(k.outcome, phases[p]), need)
                    need -= got
            # same phase, other outcomes
            for o in ["win", "loss", "draw"]:
                if need <= 0:
                    break
                if o != k.outcome:
                    got = take_from(BucketKey(o, k.phase), need)
                    need -= got
            # global pool
            if need > 0:
                flat = [i for _, lst in self._bucket_indices.items() for i in lst]
                tries = 0
                while need > 0 and flat and tries < max(1, need * 10):
                    idx = self.rng.choice(flat)
                    tries += 1
                    if idx not in used:
                        used.add(idx)
                        picked.append(idx)
                        need -= 1
                        _record(idx)

        for k, need in desired.items():
            got = take_from(k, need)
            if got < need:
                fallback_fill(k, need - got)

        return picked
