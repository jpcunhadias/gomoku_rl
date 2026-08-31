# XAI layer: Captum + SHAP attribution on Cycle 1 vs Cycle 2

**Status**: Second pass complete (scaled-up capture, cross-checked with SHAP). Produced a real,
independently-corroborated finding about Cycle 2's value head, on top of the first pass's
methodological result. Still not a closed investigation.

## Goal

The tau/dirichlet sweep was closed as inconclusive (`docs/current/SWEEP_TAU_DIRICHLET.md`). The
one result in the project that's fully solid — reconfirmed under two different arena regimes —
is the Layer 1 headline: Cycle 2 (one real iteration of self-play + retraining past a cold
start) beats Cycle 1 decisively (199-0-1). This layer asks *why*, using attribution methods on
both networks' policy and value heads, on real board positions, cross-checked across two
independent methods so no single tool's output is trusted on its own — directly applying this
project's own arena-confound lesson to a new kind of analysis.

## Method

`scripts/arena.py` has never logged move sequences, only aggregate win/loss/draw stats, so no
position from the actual, already-completed headline arena could be recovered. Rather than
inventing a parallel game-playing implementation, `scripts/arena.py::play_one()` got one small,
additive change: an optional `history` list that records every move played (default `None`,
zero effect on existing callers/tests). `scripts/xai_attributions.py` reuses `load_player()` and
`play_one()` unchanged to play real games between Cycle 1 and Cycle 2 under the exact conditions
that produced the reconfirmed headline result (schedule fix applied both sides, same stochastic
eval config: root epsilon 0.12, tau0 0.08, tau1 0.05), then picks several labeled positions per
category (opening, midgame, pre-winning-move, draw) from those games.

For each position, for each checkpoint, two independent attribution methods:
- **Captum Integrated Gradients**, against a baseline with both stone planes zeroed (the
  constant turn-indicator plane is left matching the real input, since an all-zero turn plane is
  a state the network never actually sees).
- **SHAP GradientExplainer**, against a background of ~20 real board states sampled from other
  captured positions (a genuine reference *distribution*, not a single zero baseline).

Both attribute the same fixed target per position — the logit of that network's own top *legal*
move (policy), and the scalar value output (value) — so the two methods' outputs are directly,
apples-to-apples comparable via cosine similarity over the flattened own-stone/opponent-stone
attribution channels (channel 2, the constant plane, is excluded from this comparison — see
below for why).

## A capture-methodology finding: "40 games" was really 7

The first scaled-up run (40 games) produced attribution summaries that were suspiciously
identical across supposedly different games — checked directly, and confirmed: **40 captured
games collapsed into only 7 byte-identical move sequences.** Not a bug in the new code — a
previously-unmeasured consequence of an already-documented, deliberate arena design choice
(`docs/current/SWEEP_TAU_DIRICHLET.md`, issue #2): `play_one`'s stochastic eval only randomizes
plies 0-1 (root Dirichlet noise + a small temperature); every ply after that runs at
`temperature=0`, i.e. plain deterministic MCTS argmax. Since MCTS given a fixed board, fixed
weights, and fixed simulation count is a deterministic function, **once two games land on the
same ply-0/1 outcome, their entire remainder is identical by construction** — and for these
checkpoints, the network's own prior is dominant enough that the modest root noise/temperature
rarely flips the top choice, so only a handful of distinct ply-0/1 outcomes ever actually occur.

This doesn't change the direction of any already-confirmed arena result — both sides of every
comparison get the identical protocol, so win/loss conclusions stand — but it does mean the
*effective* number of independent trials behind any arena win-rate or confidence interval may be
smaller than the raw game count suggests, since a genuinely-independent trial requires a
genuinely-distinct trajectory. **Worth checking separately whether this affects the precision
claimed for past arena Wilson CIs — not investigated here, flagged as a follow-up.**

For this script specifically: `dedupe_games()` now collapses exact-duplicate move sequences
before position selection and before building the SHAP background pool, so "N examples" means N
*distinct* positions. Second run: 40 games captured, 7 genuinely distinct, yielding 5 decisive
examples (one from each of 5 unique games) and 1 draw example (only one unique draw trajectory
was captured in this batch — thin, flagged below).

## Finding 1 (unchanged from the first pass): ply 0 is attribution-degenerate by construction

The empty-board position produced an attribution map that's exactly zero (to numerical
precision) for both stone-plane channels, for both checkpoints, under Integrated Gradients. Not
a null result about the networks — a property of the baseline choice: on an empty board, the
input's stone planes *are* all-zero, identical to the IG baseline, so there's nothing to
attribute. Whatever makes one network prefer one opening square lives entirely in the learned
weights, not in any per-cell input signal. **Per-cell input attribution cannot explain first-move
preference by construction**, for any network, under any baseline matching the empty board.

## Finding 2 (new, the headline result of this pass): Cycle 2's value head disagrees with itself across methods more than Cycle 1's does

Averaged IG-vs-SHAP cosine similarity, by category (n=5 for midgame/pre-winning-move, n=1 for
draw — ply1 excluded, see caveats):

| Category | Value cosine — Cycle 1 | Value cosine — Cycle 2 | Policy cosine — Cycle 1 | Policy cosine — Cycle 2 |
|---|---|---|---|---|
| midgame | 0.769 | **0.509** | 0.792 | 0.736 |
| pre_winning_move | 0.726 | **0.632** | 0.748 | 0.794 |
| draw (n=1) | 0.823 | **0.570** | 0.757 | 0.749 |

**Cycle 2's value head shows lower agreement between Integrated Gradients and SHAP than Cycle
1's does, in every single category with more than one example — a consistent direction, not a
coin flip.** The policy head shows no such consistent pattern (Cycle 2 is sometimes higher,
sometimes lower). This is specifically a value-head effect.

This lines up with an already-documented, completely independent finding from months earlier
(`docs/archive/CYCLE2_V4_VALIDATION_AND_ARENA.md`): Cycle 2's value head is measurably
overconfident — pre-tanh saturation jumped to 43.8% (from Cycle 1's 10.5%), Brier score degraded
to 0.677 (from 0.609), ECE to 0.226 (from 0.136). Two unrelated measurement techniques, taken
months apart, both point the same direction: **a value head that pushes toward confident
extremes also produces attributions that two independent explanation methods characterize
differently.** A plausible mechanism: an overconfident head relies on sharper, more brittle
input-output relationships (small input changes near a saturated tanh output move the gradient
landscape a lot), which two different attribution algorithms are more likely to disagree about
than a smoother, better-calibrated function would produce. Not proven here, but two independently
arrived-at signals now agree on the same direction — worth taking seriously, and worth targeting
directly if the value-head overconfidence issue (still "flagged not fixed" in the Cycle 2
report) is ever revisited.

## Caveats

- **The draw category is n=1** — only one genuinely distinct drawn trajectory was captured in
  this batch. Its number is illustrative, not a pattern, unlike the midgame/pre-winning-move
  rows above (which do show a consistent direction across 5 independent games each).
- **Ply 1 is excluded from the summary table** because both methods show weak, noisy agreement
  there for both checkpoints (roughly 0.0-0.4, no clear separation) — consistent with it being
  close to the same degenerate-baseline problem as ply 0 (only one stone differs from the IG
  baseline, and SHAP's background is comparatively much richer, so the two methods are working
  from very different amounts of signal at that ply specifically).
- **The IG baseline is a design choice, not a derived one.** A zero-stones baseline is reasonable
  but not unique; a different choice (e.g. a contrastive baseline against the second-best legal
  move) could tell a different story, especially at low ply where the current baseline is weak.
- **This whole layer rests on one capture batch.** 7 unique trajectories out of 40 games is a
  small effective sample for a method whose entire premise ("stochastic eval") is supposed to
  produce many independent trials — see the capture-methodology finding above.

## Next steps (not started)

- Check whether the "40 games, 7 unique" finding also affects the confidence claimed for past
  arena Wilson CIs (headline, sweep reruns) — a separate, potentially significant investigation
  in its own right, not done here.
- Capture more raw games specifically to grow the number of *unique* trajectories (not just the
  raw game count), especially for the draw category.
- Try a non-degenerate baseline for early-ply positions (ply 0-1) so first-move preference and
  the ply-1 method-agreement gap aren't systematically unexplainable by construction.
- If the value-head overconfidence issue is ever revisited, check whether fixing it also closes
  the IG-vs-SHAP agreement gap found here — that would be a strong confirmation the mechanism
  proposed above is the right one.
