# Arena trajectory independence: confidence intervals were overstated

**Status**: Finding confirmed at real (800 sims) headline settings. Direction of every past
arena result is unaffected; stated precision (Wilson CIs) needs a caveat everywhere.
`scripts/arena.py` now reports the corrected numbers automatically going forward — see below.

## What was found

`play_one`'s stochastic eval only randomizes plies 0-1 (root Dirichlet noise + a small
temperature); every ply after that runs at `temperature=0`, i.e. plain deterministic MCTS
argmax. MCTS given a fixed board, fixed weights, and a fixed simulation count is a deterministic
function — so **once two games land on the same ply-0/1 outcome, their entire remainder is
byte-identical.** Games sharing that outcome are not independent trials, even though each comes
from its own `play_one()` call with its own root-noise draw: they're the same trial, replayed.

This surfaced first as a side effect of the XAI capture work
(`docs/current/XAI_LAYER.md`) and was then checked directly and deliberately:

| Batch | Sims | Games | Unique trajectories | Rate |
|---|---|---|---|---|
| XAI capture | 400 | 40 | 7 | 17.5% |
| Direct check, real headline config | **800** | 60 | **3** | **5%** |

**More simulations make it worse, not better.** More search lets MCTS converge more decisively
onto whatever the dominant choice already is, so the modest root noise/temperature is even less
likely to flip the outcome. At the actual settings used for every real arena result in this
project (800 sims, root epsilon 0.12, tau0 0.08, tau1 0.05), only 1 of 30 candidate-black games
and 2 of 30 candidate-white games were genuinely distinct.

## Why this matters

A Wilson confidence interval computed as if every game were an independent Bernoulli trial is
only honest if the trials really are independent. They aren't, here. Recomputing over just the
unique trajectories in the 800-sims check:

- Nominal (60 games as 60 trials): 100% decisive winrate, Wilson 95% CI **[93.9%, 100%]**
- Effective (2 genuinely independent decisive trials): 100% decisive winrate, Wilson 95% CI
  **[34.2%, 100%]**

That's the same point estimate with a dramatically different honest uncertainty. **This applies
to every stochastic-eval arena result in the project's history**, including the headline
Cycle 1 vs Cycle 2 result (`docs/archive/CYCLE2_V4_VALIDATION_AND_ARENA.md`, and its rerun
under the fixed arena, `docs/current/SWEEP_TAU_DIRICHLET.md`) — both used the identical protocol
on the identical checkpoints, so the true number of independent trials behind the reported
199-0-1 / Wilson CI [0.981, 1.0] is almost certainly a small handful, not 200. Not rerun directly
against the full 200-game headline batch (not needed — the mechanism is a property of the
protocol and these checkpoints, already confirmed at the real settings above; rerunning larger
would refine the number but not the qualitative conclusion).

## What is *not* in question

**Every unique trajectory found across both checks — 10 in total — has Cycle 2 winning or
drawing. Cycle 1 has never won a single game, under any noise draw actually observed.** The
*direction* of every past arena result stands. What's overstated is the *precision* claimed for
it, not the conclusion. Treat every past arena Wilson CI in this project's docs as "directionally
right, narrower than the true uncertainty" rather than as a rigorous statistical guarantee.

## Fixed going forward

`scripts/arena.py` now records every game's move sequence and reports a "TRAJECTORY
INDEPENDENCE CHECK" automatically: both the nominal and an effective (deduped) Wilson CI, printed
and in the JSON output (`unique_trajectories_cand_black/white`, `unique_wilson95_lo/hi`, etc.).
Every future arena run will self-report its honest effective sample size — no more silent
overstatement.

## Not done / open

- Not rerun against the full 200-game headline batch specifically — the mechanism is now
  confirmed at real settings on a smaller batch; a full rerun would only refine the exact
  unique-trajectory count, not change the qualitative caveat above.
- Whether widening the stochastic-eval window (randomizing more than just plies 0-1) would
  produce genuinely more independent trials without changing what's actually being measured —
  not investigated. Would require touching `play_one`'s stochastic-eval design itself, which is
  now a shared, load-bearing piece of methodology across arena, the sweep, and XAI — any change
  there should be deliberate, not incidental.
