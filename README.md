# Gomoku RL

AlphaZero-style reinforcement learning implementation for Gomoku (Five-in-a-Row), featuring Monte Carlo Tree Search (MCTS) with a dual-headed deep neural network for policy and value predictions.

## Overview

This project implements a self-play training pipeline for Gomoku using:
- **Monte Carlo Tree Search (MCTS)** with configurable exploration parameters
- **Dual-headed Neural Network** (ResNet-based) for policy and value estimation
- **Self-play pipeline** with replay buffer and stratified sampling
- **Cycle-based training** with arena evaluation between model generations
- **Advanced exploration techniques** including dynamic temperature scheduling, Dirichlet noise, and opening variety

## Features

- 8×8 Gomoku board with 5-in-a-row win condition
- AlphaZero-style self-play reinforcement learning
- MCTS with dynamic c_puct scheduling and phase-based simulation budgets
- ResNet architecture with policy and value heads
- Opening variety strategies to prevent repetitive play
- Comprehensive logging and analysis tools
- Arena for model comparison and evaluation
- GPU-accelerated training with PyTorch

## Project Structure

```
gomoku_rl/
├── cli/              # Command-line interfaces
│   ├── self_play/    # Self-play game generation
│   ├── train/        # Training loop
│   ├── eval/         # Evaluation utilities
│   └── play/         # Human play interface
├── game/             # Game logic
│   ├── gomoku.py     # Board representation and rules
│   ├── encoder.py    # State encoding for neural network
│   └── player.py     # Player implementations
├── mcts/             # Monte Carlo Tree Search
│   ├── mcts.py       # MCTS algorithm
│   ├── tree_node.py  # MCTS tree node
│   └── evaluators.py # Position evaluation
├── model/            # Neural network architecture
│   └── policy_value_net.py  # ResNet-based policy-value network
├── train/            # Training components
│   ├── self_play.py  # Self-play pipeline
│   ├── train_loop.py # Training loop
│   ├── replay_buffer.py  # Experience replay
│   ├── config.py     # Configuration management
│   └── ...           # Supporting utilities
├── configs/          # Configuration files
│   └── phaseC_c1.py  # Current training configuration
├── checkpoints/      # Model checkpoints and replay buffers
├── scripts/          # Utility scripts
│   └── arena.py      # Model comparison arena
└── tests/            # Unit tests
```

## Installation

### Requirements

- [`uv`](https://docs.astral.sh/uv/) — this project is managed with `uv`, not raw `pip`/`venv`
- Python 3.12 or 3.13 (pinned via `.python-version`; `uv` will fetch it automatically)
- A CUDA-capable GPU is recommended for actual training runs — see "Where training runs" below

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd gomoku_rl

# Installs the pinned Python version, creates .venv, installs all dependencies
uv sync
```

There's no `requirements.txt` or manually-activated `venv` anymore — every command below runs
through `uv run ...` (or `make ...`, which already does this for you), so the right interpreter
and dependencies are always used without activating anything by hand.

### Where training runs

Self-play and training are **not run on a laptop**. They run on a home server over SSH
(`home-lan` on the LAN, `home-vpn` over Tailscale — see the SSH config). This Mac is for
editing code, running the test suite (`make test`), and reading/writing docs. Before running
anything real, make sure the server's checkout of this repo is on the same branch and has run
`uv sync` too.

### Orientation

Not sure where the project currently stands? Start at [`docs/README.md`](docs/README.md), then
[`docs/current/`](docs/current/) for what's actively in progress and
[`docs/CHANGELOG.md`](docs/CHANGELOG.md) for how it got there.

## Usage

The project uses a cycle-based training workflow controlled via Makefile commands.

### Quick Start

```bash
# View all available commands
make help

# Run self-play to generate training data (Cycle 1)
make self-play CYCLE=1

# Train the model on collected data
make train CYCLE=1

# Evaluate model performance
make arena CANDIDATE_CYCLE=1 BASELINE_CYCLE=0

# Analyze self-play statistics
make analyze CYCLE=1
```

### Training Pipeline

A typical training cycle consists of:

1. **Self-play**: Generate games using MCTS with the current model
   ```bash
   make self-play CYCLE=1
   ```

2. **Training**: Train the model on collected experiences
   ```bash
   make train CYCLE=1
   ```

3. **Evaluation**: Compare against previous model in arena
   ```bash
   make arena CANDIDATE_CYCLE=1 BASELINE_CYCLE=0
   ```

4. **Debug** (optional): Validate model and training quality
   ```bash
   make debug CYCLE=1
   ```

### Advanced Configuration

Override configuration parameters at runtime:

```bash
# Custom learning rate
make train CYCLE=2 ARGS="--learning_rate 0.0005"

# Custom number of self-play games
make self-play CYCLE=2 ARGS="--num_self_play_games 300"

# Custom arena settings
make arena CANDIDATE_CYCLE=2 BASELINE_CYCLE=1 ARGS="--games 400 --sims 1000"
```

### Configuration Files

Edit `configs/phaseC_c1.py` to modify:
- Self-play parameters (games, simulations)
- Training hyperparameters (learning rate, batch size, epochs)
- MCTS settings (c_puct, temperature, Dirichlet noise)
- Opening variety strategies
- Evaluation criteria

## Key Components

### MCTS Configuration

- **Dynamic c_puct scheduling**: Adaptive exploration based on move number
- **Phase-based simulation budgets**: Different search depths for opening/mid/endgame
- **Temperature scheduling**: Controls move selection randomness
- **Dirichlet noise**: Adds exploration at root node

### Neural Network

- **Architecture**: ResNet with 5 residual blocks
- **Input**: 3-channel board representation (current player, opponent, empty)
- **Policy head**: Outputs move probabilities (64 logits for 8×8 board)
- **Value head**: Outputs win probability estimate [-1, 1]

### Replay Buffer

- **Stratified sampling**: Balances samples across game phases
- **Data augmentation**: 8-fold symmetry (rotations + reflections)
- **Position canonicalization**: Consistent representation regardless of player

## Code Quality

```bash
# Lint code
make lint

# Auto-fix linting issues
make lint-fix

# Format code
make format

# Lint-fix + format together
make all
```

As of this writing `make lint` reports ~250 pre-existing issues (mostly import ordering and
`pyupgrade` hints) from before `ruff` was wired into `uv`/pre-commit — none block tests or
training, but `make lint-fix` hasn't been run repo-wide yet since that's a separate, larger
cleanup than this pass covered.

## Testing

```bash
# Run all tests
make test          # or: uv run pytest

# Run a specific test file
uv run pytest tests/test_mcts.py

# Run with coverage
uv run pytest --cov=.
```

## Checkpoints and Artifacts

The project saves artifacts in structured directories:

- `checkpoints/models/` - Model checkpoints (c1_cycleN_last.pth)
- `checkpoints/buffers/` - Replay buffers (replay_c1_cycleN.pkl)
- `checkpoints/arena/` - Arena comparison results
- `logs/` - Training logs and JSONL game records

## Development Notes

- The project uses **cycle-based versioning** (c1, c2, etc.) for major configuration changes
- Each cycle maintains separate model checkpoints and replay buffers
- Arena evaluation determines if a new model should replace the baseline
- Self-play generates JSONL logs for detailed game analysis

## Performance Tips

- Use GPU for training (automatically detected by PyTorch)
- Adjust `batch_size` based on available GPU memory
- Tune `num_self_play_games` and `self_play_num_simulations` for speed/quality tradeoff
- Use `reload_buffer_every` to incrementally load large replay buffers

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

## Acknowledgments

Built on principles from DeepMind's AlphaZero research, adapted for Gomoku with custom enhancements for exploration and training stability.
