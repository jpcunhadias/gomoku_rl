# Documentation

This directory is the source of truth for what's actually been done, found, and is still open.
If you're picking this project up cold — a future session, a different agent, or just yourself
after a break — read this file first.

## Structure

- **`CHANGELOG.md`** — consolidated history of all training cycles, in chronological order
- **`current/`** — active documentation for whatever's in progress right now
- **`archive/`** — closed-out investigations and completed cycles, kept for reference

## Status as of this writing

**Read this part first: every arena result in the project to date is provisional.**
`scripts/arena.py` used to hard-code the candidate to always get a c_puct search-time
exploration bonus that the baseline never got (up to eff. c_puct 4.0 vs. a flat 1.5),
regardless of which model was actually being tested — and the model under test was always
passed as candidate, in every comparison this project has ever run. That means every
"decisive win" recorded so far (Cycle 2 over Cycle 1, every sweep point over Cycle 2, the
tau-axis's 100-0-0 results) could be partly or wholly a search-time artifact rather than
genuine trained-model strength. **Fixed** — `--candidate_schedule` now exists alongside
`--baseline_schedule`, both default to `False`, so `make arena` is fair by default — but
nothing has been rerun under the fix yet. Treat every arena number dated before this fix as
directionally suggestive, not confirmed. A background sweep run in progress when this was
found was deliberately stopped mid-run rather than let it keep spending compute on the
confounded setup.

**Layer 1 (Cycle 1 → Cycle 2) is done, modulo the caveat above.** Two real training
generations exist: Cycle 1 (cold-start, trained on schedule bugs found and fixed) and Cycle 2
(v4 exploration config). See `archive/CYCLE1_COLDSTART_MECHANISM.md` and
`archive/CYCLE2_V4_VALIDATION_AND_ARENA.md` — both good for the mechanism findings, not yet
reconfirmed for the exact arena numbers.

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

**The tau/dirichlet_epsilon sweep (Layer 2) is paused, pending a rerun under the arena fix.**
The tau axis produced a very clean-looking result (4 points, each beating the previous
decisively) — worth reconfirming now that the schedule confound is fixed, not assuming it
survives unchanged. The dirichlet axis has one point done (44, dirichlet 0.75x — also
confounded, not yet rerun) and one not started (46). The Cycle 1 vs 2 headline arena rerun
never got to run before the job was stopped.

**Code quality**: a full pass closed ~250 lint findings (most mechanical) and fixed a real
divide-by-zero in MCTS's visit-count normalization (silent NaN probabilities if every root
child ended up with 0 visits — now has a dedicated regression test) and a scattering of
`zip()` calls without `strict=` (verified safe to make strict — genuine invariant checks, not
just satisfying the linter). Remaining, left as documented debt: line-length and a handful of
`sys.path`-before-import lines in standalone debug scripts (a deliberate pattern, not a bug).

**Still open / not started**:
- Rerunning the sweep (and the Cycle 1 vs 2 headline arena) under the fixed, symmetric arena
  — the natural next step, not yet done.
- DVC/MLflow backup wiring — the server's disk is still the only copy of everything. Parked
  since early in the project; worth revisiting given how much has been generated since.
- The XAI layer (captum/shap) — planned next step once arena results are trustworthy again.
- Whether the sweep's conclusions hold at full compute (200 games/800 sims, not the reduced
  100/400 used for speed) — untested either way.
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
