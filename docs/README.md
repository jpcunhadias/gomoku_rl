# Documentation

This directory is the source of truth for what's actually been done, found, and is still open.
If you're picking this project up cold — a future session, a different agent, or just yourself
after a break — read this file first.

## Structure

- **`CHANGELOG.md`** — consolidated history of all training cycles, in chronological order
- **`current/`** — active documentation for whatever's in progress right now
- **`archive/`** — closed-out investigations and completed cycles, kept for reference

## Status as of this writing

**Layer 1 (Cycle 1 → Cycle 2) is done.** Two real training generations exist: Cycle 1 (cold-start,
trained on schedule bugs found and fixed) and Cycle 2 (v4 exploration config, validated). See
`archive/CYCLE1_COLDSTART_MECHANISM.md` and `archive/CYCLE2_V4_VALIDATION_AND_ARENA.md`.
**Caveat**: the headline Cycle 1 vs 2 arena result in that doc predates the arena-determinism fix
below — a rerun is in progress (see "In progress" below); treat the exact numbers there as
directionally right but not yet reconfirmed.

**The value-head "overconfidence" scare is closed, and it was a false alarm.** Every calibration
check up to that point compared a model against *its own* training buffer — never a fair test.
`scripts/diagnose_value_head_holdout.py` (a real held-out split, first one in the project) showed
training actually *improves* held-out calibration. See `archive/VALUE_HEAD_CALIBRATION_INVESTIGATION.md`.

**Three real methodology bugs were found and fixed**, all of which quietly affected results
before being caught:
1. Arena was fully deterministic (no `--stochastic_eval`, no opening variety) — every arena
   result in the project was really 2 unique games replayed N times, not N independent trials.
   Fixed: `make arena` now runs real independent games by default.
2. The entropy check pooled all plies into one misleading number, and separately, Cycle 2's own
   buffer turned out to be a poor comparison baseline (25% of it was inherited cold-start data
   from Cycle 1, quietly skewing its own ply 1/2 entropy readings).
3. Three pipeline soundness gaps: the test suite was writing real files into `checkpoints/`
   (tests thought a config field sandboxed them; it wasn't read); "best checkpoint" was selected
   by training-set loss, which structurally can't detect overfitting; and the held-out calibration
   check existed only as a script you had to remember to run, not part of `make debug`. All three
   fixed — see the `docs/CHANGELOG.md` entries around the sweep.

Full story, including how each was found: `current/SWEEP_TAU_DIRICHLET.md`.

**The tau/dirichlet_epsilon sweep (Layer 2) is in progress.** The tau axis is done and conclusive:
4 points (0.75x, 1.0x-clean, 1.25x, 1.5x), each stronger setting beating the previous one
decisively (two of them 100-0-0, every game). The dirichlet axis (0.75x, 1.25x) and a rerun of
the original Cycle 1 vs 2 headline arena (with the determinism fix) were both **still running
in the background as of this writing** — check `current/SWEEP_TAU_DIRICHLET.md` for the design
and whatever results have landed, and the server's `logs/dirichlet_and_headline_rerun.log` for
live progress if you have access.

**Code quality**: a full pass closed ~250 lint findings (most were mechanical — type-hint
modernization, import sorting) and fixed two real bugs found along the way: a divide-by-zero in
MCTS's visit-count normalization (silent NaN probabilities if every root child ended up with 0
visits) and a scattering of `zip()` calls without `strict=`, each verified safe to make strict
(all pairs are genuinely equal-length by construction — this was a real invariant check, not
just satisfying the linter). Remaining, left as documented debt: line-length and a handful of
`sys.path`-before-import lines in standalone debug scripts (a deliberate pattern, not a bug).

**Still open / not started**:
- DVC/MLflow backup wiring — the server's disk is still the only copy of everything. Parked
  since early in the project; worth revisiting given how much has been generated since.
- The XAI layer (captum/shap) — planned next step once the above settles, not yet begun.
- Consolidating the sweep results into `CHANGELOG.md`/`archive/` once the background run finishes.

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
