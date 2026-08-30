# Gomoku RL

AlphaZero-style self-play reinforcement learning for 8×8 Gomoku (five-in-a-row): MCTS search
guided by a dual-head (policy + value) ResNet, trained through cycle-based self-play, with a
parameter-sweep methodology and calibration/evaluation tooling built around it.

## Status

This is an active research project, not a finished package. For what's actually been done,
found, and is still open — read **[`docs/README.md`](docs/README.md) first**, before this file
or the code. It is kept up to date; this README is not the place for project status.

## Requirements

- [`uv`](https://docs.astral.sh/uv/) — this project is managed with `uv`, not raw `pip`/`venv`
- Python 3.12 or 3.13 (pinned via `.python-version`; `uv` fetches it automatically)
- A CUDA-capable GPU for real training runs — see "Where training runs" below

## Setup

```bash
git clone <repository-url>
cd gomoku_rl
uv sync   # pins the Python version, creates .venv, installs all dependencies
```

No `requirements.txt` or manually-activated `venv` — every command below runs through
`uv run ...` (or `make ...`, which already does this), so the right interpreter and dependencies
are always used without activating anything by hand.

## Where training runs

Self-play and training run on a home server over SSH (`home-lan` on the LAN, `home-vpn` over
Tailscale), not on a laptop. A laptop checkout is for editing code, running the test suite
(`make test`), and reading/writing docs. Before running anything real, confirm the server's
checkout is on the same branch and has run `uv sync` too.

## Project structure

```
gomoku_rl/
├── cli/              # Entry points: self-play, train, eval, human play
├── game/             # Board representation, rules, state encoding
├── mcts/             # Monte Carlo Tree Search (search, tree node, evaluators)
├── model/            # PolicyValueNet (ResNet backbone, policy + value heads)
├── train/            # Self-play runner, training loop, replay buffer, config loading
├── configs/          # One file per training config (see "Configuration files" below)
├── scripts/          # arena.py, held-out calibration check, buffer/exploration analysis
├── debug/            # Per-checkpoint diagnostics: value/policy head, MCTS target entropy
├── utils/            # Cycle-aware path resolution, seeding
├── tests/            # pytest suite
├── docs/             # Project status, changelog, investigation write-ups (start here)
└── checkpoints/      # Models, buffers, arena results (gitignored, server-only)
```

## Usage

The project runs as a cycle-based workflow through `make`. `make help` lists everything; the
core loop for a cycle `N`:

```bash
make self-play CYCLE=N [CONFIG=<name>]     # generate self-play games into a buffer
make train CYCLE=N [CONFIG=<name>]         # train on that buffer
make debug CYCLE=N                         # value/policy head checks + held-out calibration
make arena CANDIDATE_CYCLE=N BASELINE_CYCLE=M   # compare two cycles' trained models
```

`CONFIG` defaults by cycle number (see the `ifeq` block at the top of the `Makefile`); pass it
explicitly for anything other than the two default configs. Override any config field at the
command line:

```bash
make train CYCLE=2 ARGS="--learning_rate 0.0005"
make self-play CYCLE=2 ARGS="--num_self_play_games 300"
make arena CANDIDATE_CYCLE=2 BASELINE_CYCLE=1 ARGS="--games 100 --sims 400"
```

`make arena` runs with `--stochastic_eval` by default (real independent games, not one
deterministic game replayed N times — see `docs/README.md` for why that distinction matters).

## Configuration files

Each file in `configs/` is a self-contained training config (a `get_config()` returning a
`SimpleNamespace`) — self-play parameters, training hyperparameters, and MCTS/exploration
settings (`c_puct`, temperature schedule, Dirichlet noise). `phaseC_c1.py`/`phaseC_c2.py` are
the two real training cycles; `sweep_*.py` are parameter-sweep points, each varying exactly one
setting from a documented baseline (see `docs/current/` for the active sweep's design). New
sweep points or configs should follow that same one-file-per-config, one-change-at-a-time pattern.

## Code quality

```bash
make lint        # ruff check
make lint-fix     # ruff check --fix
make format       # ruff format
make all          # lint-fix + format
```

## Testing

```bash
make test                          # or: uv run pytest
uv run pytest tests/test_mcts.py   # a single file
uv run pytest --cov=.              # with coverage
```

## Checkpoints and artifacts

Server-only, gitignored:

- `checkpoints/models/c1_cycle{N}_{best,last}.pth`
- `checkpoints/buffers/replay_c1_cycle{N}.pkl`
- `checkpoints/arena/arena_c{X}_vs_c{Y}.json`
- `logs/` — training logs and JSONL game records
