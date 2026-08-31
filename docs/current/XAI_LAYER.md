# XAI layer: Captum attribution on Cycle 1 vs Cycle 2

**Status**: First pass complete. Method validated, produced real, interpretable, if
small-sample, findings. Not a closed investigation — this is a starting point for deeper work,
not a final report.

## Goal

The tau/dirichlet sweep was closed as inconclusive (`docs/current/SWEEP_TAU_DIRICHLET.md`). The
one result in the project that's fully solid — reconfirmed under two different arena regimes —
is the Layer 1 headline: Cycle 2 (one real iteration of self-play + retraining past a cold
start) beats Cycle 1 decisively (199-0-1). This layer asks *why*, using Captum attribution
(Integrated Gradients) on both networks' policy and value heads, on real board positions.

## Method

`scripts/arena.py` has never logged move sequences, only aggregate win/loss/draw stats, so no
position from the actual, already-completed headline arena could be recovered. Rather than
inventing a parallel game-playing implementation, `scripts/arena.py::play_one()` got one small,
additive change: an optional `history` list that records every move played (default `None`,
zero effect on existing callers/tests). `scripts/xai_attributions.py` reuses `load_player()` and
`play_one()` unchanged to play real games between Cycle 1 and Cycle 2 under the exact conditions
that produced the reconfirmed headline result (schedule fix applied both sides, same stochastic
eval config: root epsilon 0.12, tau0 0.08, tau1 0.05), then picks a small, labeled set of real
positions from those games — not an arbitrary or hand-picked board.

For each position, for each checkpoint: Integrated Gradients attributes (a) the logit of that
network's own top *legal* move, and (b) the scalar value output, against a baseline with both
stone planes zeroed (the constant turn-indicator plane is left matching the real input, since an
all-zero turn plane is a state the network never actually sees).

First run: 10 games (5 pairs), 400 sims/move, seed 42. `checkpoints/xai/games.json` holds the
captured move lists; `checkpoints/xai/attributions.json` and `checkpoints/xai/plots/` hold the
results (both gitignored, like everything under `checkpoints/`, but reproducible by rerunning
the script — or by rerunning attribution alone against `games.json` via `--positions_file`,
without replaying the expensive MCTS games again).

## A methodological finding first: ply 0 is attribution-degenerate by construction

The empty-board position (`ply0_empty`) produced an attribution map that's exactly zero (to
numerical precision) for both stone-plane channels, for both checkpoints. This isn't a null
result about the networks — it's a property of the baseline choice. On an empty board, the
input's stone planes *are* all-zero, identical to the baseline. Integrated Gradients integrates
gradient × (input − baseline) along a path between the two; where they're identical, there is
nothing to attribute. Whatever makes one network prefer one opening square over another lives
entirely in the learned convolutional/fully-connected weights, not in any per-cell input signal
— there's no input signal to point to yet. **Per-cell input attribution cannot explain first-move
preference by construction, for any network, under any baseline that matches the empty board.**
Worth remembering before reading too much into an "uninformative" attribution map at low ply
counts — check whether the position is degenerate before concluding the network's decision has
no explanation.

## Real findings (n=1 per case — illustrative, not yet statistically established)

**Ply 1 (one stone on the board) shows Integrated Gradients working exactly as expected, and a
real quantitative difference between the two networks.** With exactly one differing cell between
input and baseline, both networks' policy attribution is (correctly) concentrated entirely on
that one cell — nothing spread elsewhere, confirming the method is behaving correctly, not just
producing plausible-looking noise. But the *magnitude* differs sharply: Cycle 1 attributes a much
larger effect to that single opposing stone (~1.7-1.8) than Cycle 2 does (~0.5-0.6) when
justifying its own top move. Read plainly: Cycle 1's early move choice is more reactive to the
one visible fact on the board; Cycle 2 leans more on whatever it learned about board structure in
general, and less on that single piece of local information. Consistent with Cycle 1 being the
cold-start model with comparatively little learned structure to fall back on.

**At a dense pre-winning-move position, Cycle 2's value-head attribution is sharply localized on
a specific stone cluster (a near-complete line); Cycle 1's is diffuse across many cells.** Same
board, same task (explain the value estimate), very different internal story: Cycle 2 appears to
have learned to recognize a concrete tactical pattern and weight it heavily, while Cycle 1
spreads its attention thinly without singling out the pattern that actually decides the position.
This is a plausible, mechanistically-grounded account of *why* Cycle 2 plays better, not just
*that* it does — exactly the kind of thing arena win/loss counts alone can't show.

**On a position from a drawn game, the two networks disagree sharply — not just in magnitude but
in sign.** Cycle 1 evaluates the position at value +0.431 (favorable for the player to move);
Cycle 2 evaluates the *same* position at −0.997 (near-certain loss). The attribution maps differ
correspondingly, including opposite-signed attribution on the same board region. Since the game
was drawn, neither prediction was simply "right," but the sharpness of Cycle 2's read (near ±1,
where Cycle 1 hedges close to zero) is consistent with the documented arena finding that Cycle 2
wins outright as Black but only ever draws as White — Cycle 2 reads positions more decisively,
for better or worse, than the noisier Cycle 1.

## Caveats

- **n=1 per finding.** Each observation above comes from a single captured position in a single
  category, from one small batch of 10 games. These are illustrative, mechanistically plausible
  readings, not statistically established claims — treat them the same way the sweep's early
  100-0-0 results should have been treated the first time: as worth investigating further, not
  as settled.
- **Baseline choice matters and was chosen, not derived.** A zero-stones baseline is a reasonable
  default but not the only valid one; different baselines (e.g. averaging over random noise, or
  a contrastive baseline against the second-best legal move) could tell a different, possibly
  more informative story, especially for low-ply positions where the current baseline is weak or
  degenerate (see above).
- **10 games is a small, one-off capture**, not reconfirmed across multiple batches the way the
  arena headline was. Worth a larger, repeated capture before treating any single position's
  story as representative rather than anecdotal.

## Next steps (not started)

- Scale up: more captured games, more positions per category, to see whether the ply-1
  reactivity gap and the pre-winning-move localization difference hold up as *patterns* across
  many positions, not just the one example each above.
- Try a non-degenerate baseline for early-ply positions (ply 0-2) so first-move and early-game
  preferences aren't systematically unexplainable by construction.
- Extend from Integrated Gradients to a second method (e.g. SHAP, also already a project
  dependency) as a cross-check — the sweep's own lesson was not to trust a single method's
  result without independent confirmation.
