# Cycle 2 — v4: Written, Not Yet Run

**Status**: This is the actual frontier of the project. Everything else in `docs/archive/`
already happened; this has not.

## What v4 is

A rebalance of Cycle 2's exploration knobs, sitting between v2 (too uniform — normalized
entropy 0.973) and v3 (too sharp — normalized entropy 0.196, raw entropy 0.805). See
`docs/CHANGELOG.md` ("Phase 4") for the full v2→v3→v4 story and the exact parameter deltas.

The config is already written: `configs/phaseC_c2.py`.

## What's missing

v3 was run and measured, but no debug report was ever committed for it — the only record of
its numbers is a code comment in `phaseC_c2.py`'s docstring. v4 has not been run at all.

## Next actions (on the server, not this Mac)

```bash
# 1. Regenerate Cycle 2 self-play with v4
make self-play CYCLE=2

# 2. Check MCTS target entropy BEFORE training on it — this is the check that caught
#    the v2 problem in the first place. Don't skip it.
uv run python debug/check_mcts_target_entropy.py
# Expect: normalized entropy in [0.45, 0.65] AND raw entropy >2.0, simultaneously.

# 3. If entropy checks pass, train
make train CYCLE=2

# 4. Full debug pass (value + policy head + smoke check)
make debug CYCLE=2

# 5. Arena vs. the v2 model (or v1, if v2's checkpoint no longer exists) to see if it's
#    actually stronger, not just better-calibrated on paper
make arena CANDIDATE_CYCLE=2 BASELINE_CYCLE=1
```

## When this is done

Move this file to `docs/archive/`, add a `CYCLE2_V4_VALIDATION_REPORT.md` there with the real
numbers (mirroring the format of `CYCLE2_VALIDATION_REPORT.md`), and update `docs/CHANGELOG.md`
with the outcome — whichever way it goes.
