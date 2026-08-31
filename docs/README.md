# Documentation

This directory is the source of truth for what's actually been done, found, and is still open.
If you're picking this project up cold — a future session, a different agent, or just yourself
after a break — read this file first.

## Structure

- **`CHANGELOG.md`** — consolidated history of all training cycles, in chronological order
- **`current/`** — active documentation for whatever's in progress right now
- **`archive/`** — closed-out investigations and completed cycles, kept for reference

## Status as of this writing

**The arena confound is fixed, and the rerun is complete (2026-08-31).** `scripts/arena.py`
used to hard-code the candidate to always get a c_puct search-time exploration bonus that the
baseline never got, regardless of which model was actually being tested — and the model under
test was always passed as candidate, in every comparison this project had ever run. That's
fixed (`--candidate_schedule`/`--baseline_schedule` both default `False`, so `make arena` is
fair by default), and all six affected comparisons have now been rerun under the fix. Result,
in short: **Layer 1 (Cycle 1 → Cycle 2) survived intact** (199-0-1, as decisive as before), but
**the tau/dirichlet sweep's headline conclusion did not** — three of its four supporting
comparisons reversed outcome or dissolved to an exact tie under the fix. Full numbers and
analysis: `current/SWEEP_TAU_DIRICHLET.md`.

**Layer 1 (Cycle 1 → Cycle 2) is done and confirmed.** Two real training generations exist:
Cycle 1 (cold-start, trained on schedule bugs found and fixed) and Cycle 2 (v4 exploration
config). See `archive/CYCLE1_COLDSTART_MECHANISM.md` and
`archive/CYCLE2_V4_VALIDATION_AND_ARENA.md` for the mechanism findings; the headline arena
number itself was reconfirmed under the fixed, symmetric arena at 199 wins / 0 losses / 1 draw
for Cycle 2 (200 games, 800 sims) — this is the one fully solid, load-bearing result in the
project and the recommended basis for starting the XAI layer, rather than waiting on the sweep
below to resolve.

**The value-head "overconfidence" scare is closed, and it was a false alarm.** Every calibration
check up to that point compared a model against *its own* training buffer — never a fair test.
`scripts/diagnose_value_head_holdout.py` (a real held-out split, first one in the project) showed
training actually *improves* held-out calibration. See `archive/VALUE_HEAD_CALIBRATION_INVESTIGATION.md`.
This check now also covers the **policy** head (KL to held-out targets, normalized entropy,
top-1 agreement) — it used to be value-only, an asymmetry in scrutiny that's now fixed too.

**Methodology bugs found and fixed, in the order discovered** (full story:
`current/SWEEP_TAU_DIRICHLET.md`):
1. Sequential sweep cycle numbers collided with the buffer-seeding fallback (fixed with
   non-adjacent cycle ids).
2. Arena was fully deterministic (no `--stochastic_eval`, no opening variety) — every arena
   result was really 2 unique games replayed N times, not N independent trials. Fixed.
3. The entropy check pooled all plies into one misleading number, and Cycle 2's own buffer
   turned out to be a poor comparison baseline (25% inherited cold-start data from Cycle 1,
   quietly skewing its own ply 1/2 entropy readings). Fixed with a per-ply breakdown and a
   clean re-measured baseline.
4. Three pipeline soundness gaps: the test suite was writing real files into `checkpoints/`
   (a config field was believed to sandbox them; it was never read); "best checkpoint" was
   selected by training-set loss, which structurally can't detect overfitting; the held-out
   calibration check existed only as a script you had to remember to run. All three fixed.
5. **The arena schedule asymmetry described above** — found during a deliberate audit
   ("are we actually ready to trust these results"), not during normal development. Fixed.

**A deliberate soundness audit also verified two things that turned out fine, and closed a
test-coverage gap**:
- `train/augmentation.py`'s 6 symmetry transforms are correct — a dedicated test confirms
  state and policy stay aligned under every transform, and that all 6 are genuinely distinct
  (see `tests/test_augmentation.py`). This had never been tested before.
- A tautological self-play test (asserted `entropy >= 0`, true of any policy) was replaced
  with deterministic tests of the actual temperature→entropy mechanism
  (`tests/test_mcts_extended.py`), plus a regression test for the MCTS divide-by-zero fix
  below.

**The tau/dirichlet_epsilon sweep (Layer 2) is closed, parked as inconclusive.** The tau axis
had produced a very clean-looking result (4 points, each beating the previous decisively) —
that did not survive reconfirmation under the fixed arena. Under the fix: tau 0.75x (point 31)
now *beats* Cycle 2 (it used to lose outright); tau 1.25x (point 42) now *loses* to Cycle 2 (it
used to win outright); tau 1.5x (point 61) is now an exact tie against point 42 (it used to win
outright). Only one comparison in the whole axis reproduced identically (42 vs 50, 100-0-0 both
times). Point 44 (dirichlet 0.75x) was also rerun and is robustly weaker than the clean
baseline (0-100-0, no color dependence — the cleanest signal in the batch). Point 46 was never
rerun — it only has a `_last` checkpoint from before the original job was stopped mid-training,
not a `_best`, so it isn't rerunnable without finishing that training run first.

Several of the rerun results show total color determinism (100% win/loss or an exact split by
color) at only 50-100 games — read as non-transitive, high-variance matchups rather than a
smooth strength ordering, which a star-comparison design (each point vs. one shared reference)
at this sample size can't reliably resolve. **Recommendation: don't invest further in rescuing
this sweep** — a trustworthy ranking would need either many more games per comparison or a
round-robin design, and neither is worth doing right now. One loose end, optional and
non-blocking: 42 beats 50 100-0 but loses to Cycle 2, even though Cycle 2 and Cycle 50 share an
identical config — a direct 50-vs-2 rerun would resolve the apparent non-transitivity, but
nothing is gated on it. Full detail: `current/SWEEP_TAU_DIRICHLET.md`.

**Code quality**: a full pass closed ~250 lint findings (most mechanical) and fixed a real
divide-by-zero in MCTS's visit-count normalization (silent NaN probabilities if every root
child ended up with 0 visits — now has a dedicated regression test) and a scattering of
`zip()` calls without `strict=` (verified safe to make strict — genuine invariant checks, not
just satisfying the linter). Remaining, left as documented debt: line-length and a handful of
`sys.path`-before-import lines in standalone debug scripts (a deliberate pattern, not a bug).

**Still open / not started**:
- The XAI layer (captum/shap) — the natural next step now that arena results are trustworthy
  again. Should target the Cycle 1 vs Cycle 2 pair specifically; doesn't need the sweep below
  to resolve first.
- DVC/MLflow backup wiring — the server's disk is still the only copy of everything. Parked
  since early in the project; worth revisiting given how much has been generated since.
- Optional, non-blocking: a direct 50-vs-2 arena rerun to resolve the apparent non-transitivity
  noted in the sweep rerun (42 > 50 but 2 > 42, while 2 and 50 share an identical config).
- Whether a properly-powered redesign of the tau/dirichlet sweep (more games, or round-robin
  instead of star-comparison) is worth doing at all, now that the original design's conclusions
  didn't survive — not decided; not currently planned.
- Why discounting value targets made calibration worse during the value-head investigation —
  found, reverted, never actually explained.

## Workflow

1. **During active work**: keep docs in `current/`
2. **Once it's done**: move the doc(s) to `archive/`, update `CHANGELOG.md` with a summary entry
3. **Starting something new**: add a new doc to `current/`

## Key tools worth knowing about

- `make debug CYCLE=N` — value/policy head checks, training smoke test, and (if a previous
  cycle's checkpoint exists) a held-out calibration check, all in one command.
- `debug/check_mcts_target_entropy.py --buffer <path>` — per-ply MCTS target entropy, not just
  a pooled number. Use this over eyeballing self-play debug logs; the per-ply breakdown has
  caught real, otherwise-invisible effects (see the sweep doc).
- `scripts/diagnose_value_head_holdout.py [--quick]` — proper train/held-out split calibration
  check; `--quick` for routine use, full mode (default) for one-off investigations comparing
  training configs.
- `make arena CANDIDATE_CYCLE=X BASELINE_CYCLE=Y ARGS="--games N --sims S"` — runs with
  `--stochastic_eval` by default now, so results are real independent trials.
