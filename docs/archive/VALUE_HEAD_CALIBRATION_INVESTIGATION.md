# Value-Head "Overconfidence" Investigation

**Status**: Closed — the original diagnosis doesn't survive a controlled test.

## The original flag

`docs/archive/CYCLE2_V4_VALIDATION_AND_ARENA.md`'s Cycle 2 debug check (`make debug CYCLE=2`)
showed worse value-head calibration than Cycle 1: Brier 0.677 (was 0.609), ECE 0.226 (was
0.136), pre-tanh saturation 43.8% (was 10.5%, threshold <20%).

## Two candidate fixes tried — both failed

1. **Discount value targets to `z*0.95`** (reduce the pressure to saturate `tanh` fitting an
   exact +-1 label) + **raise value-head weight decay 2e-4 -> 1e-3**. Retrained against the
   *existing* Cycle 2 buffer (isolates training dynamics from data). Result: worse, not better —
   Brier 0.677 -> 0.794, ECE 0.226 -> 0.315.
2. **Isolated the two changes**: reverted weight decay to 2e-4, kept the target discount.
   Identical result (byte-identical loss trajectory and calibration numbers to the combined
   test) — weight decay was a complete no-op at these magnitudes under plain `torch.optim.Adam`
   (its L2-via-gradient weight decay isn't properly decoupled from Adam's adaptive per-parameter
   scaling, unlike `AdamW`). The target discount alone caused the regression, for a reason not
   fully understood (the magnitude doesn't match a simple "predictions now cap lower" story).

Both changes reverted; the original arena-validated checkpoint was restored and reverified
(Brier 0.676972, ECE 0.225687, saturation 0.438 — exact match to the pre-investigation numbers).

## The actual problem: the comparison method, not the model

Every check up to this point — including the original Cycle 1 vs. Cycle 2 comparison that
started this investigation — evaluated a model on samples drawn from **the same buffer it
trained on**. Comparing Cycle 1's calibration (on Cycle 1's buffer) to Cycle 2's calibration (on
Cycle 2's buffer) is not a controlled before/after test of the same model on the same data — it's
two different buffers with different games and different tactical composition. A "regression"
there could just mean Cycle 2's buffer is intrinsically harder to call, not that the value head
got worse at its job.

## The real test

`scripts/diagnose_value_head_holdout.py`: splits Cycle 2's buffer 90/10, trains fresh copies of
the model (from the same Cycle 1 starting checkpoint) on the 90%, evaluates Brier/ECE/saturation
on the untouched 10% — a genuine generalization check, done for the first time in this project.

| | Brier | ECE | Saturation |
|---|---|---|---|
| Cycle 1 checkpoint, untrained on this data | 0.816 | 0.178 | 7.1% |
| Trained on Cycle 2 (Adam, matches original) | **0.548** | **0.138** | 40.0% |
| Trained on Cycle 2 (AdamW) | 0.565 | 0.156 | 38.8% |

Training on Cycle 2's data **substantially improves** Brier and ECE on genuinely unseen
positions. Saturation is still elevated relative to the debug script's 20% heuristic threshold,
but since the metrics that actually measure prediction *accuracy* improved rather than degraded,
high saturation alone doesn't appear to be hurting anything here — it plausibly just reflects
that a lot of these positions are genuinely decisive. AdamW does not help (slightly worse Brier/
ECE than plain Adam), so that hypothesis doesn't hold up either.

## Conclusion

The thing that started this investigation — "Cycle 2's value head is worse than Cycle 1's" —
doesn't survive a controlled test. Closing this out rather than continuing to chase a fix for a
problem that likely doesn't exist. The one thing worth keeping from this detour:
`scripts/diagnose_value_head_holdout.py` is a real, reusable held-out calibration check the
project didn't have before — every prior debug check evaluated on training-distribution data.
Worth reaching for this instead of `debug/value_head_check.py` alone when a future cycle's
calibration needs checking, especially across different cycles/buffers.

## Note for later

The 20%-saturation pass/fail threshold in `debug/value_head_check.py`'s checkbox summary isn't
validated against anything — it's a heuristic someone picked. Worth treating saturation as a
secondary signal to check alongside Brier/ECE, not a standalone red flag.
