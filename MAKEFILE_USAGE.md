# Makefile Usage Guide

The Makefile provides convenient shortcuts for common project tasks.

## Quick Reference

### View Available Commands
```bash
make help
```

### Training & Self-Play

**Train Cycle 2** (auto-selects `phaseC_c2` config):
```bash
make train CYCLE=2
```

**Train with explicit config**:
```bash
make train CYCLE=2 CONFIG=phaseC_c2
```

**Train with custom arguments**:
```bash
make train CYCLE=2 ARGS="--learning_rate 0.0005 --batch_size 256"
```

**Run self-play**:
```bash
make self-play CYCLE=2
```

### Analysis & Evaluation

**Analyze self-play data**:
```bash
make analyze CYCLE=2
```

**Run debug checks**:
```bash
make debug CYCLE=2
```

**Arena evaluation** (compare models):
```bash
make arena CANDIDATE_CYCLE=2 BASELINE_CYCLE=1
```

### Code Quality

**Check linting**:
```bash
make lint
```

**Auto-fix linting issues**:
```bash
make lint-fix
```

**Format code**:
```bash
make format
```

**Clean checkpoints and logs**:
```bash
make clean
```

## Configuration

The Makefile automatically selects config files based on cycle:
- **Cycle 1**: Uses `phaseC_c1` by default
- **Cycle 2**: Uses `phaseC_c2` by default

You can override this by explicitly specifying `CONFIG`:
```bash
make train CYCLE=2 CONFIG=phaseC_c1  # Use c1 config even for cycle 2
```

## Examples

### Complete Cycle 2 Workflow

```bash
# 1. Run self-play
make self-play CYCLE=2

# 2. Analyze results
make analyze CYCLE=2

# 3. Train model
make train CYCLE=2

# 4. Evaluate in arena
make arena CANDIDATE_CYCLE=2 BASELINE_CYCLE=1
```

### Training with Custom Hyperparameters

```bash
make train CYCLE=2 ARGS="--learning_rate 0.0005 --batch_size 256 --epochs 15"
```

### Debugging

```bash
# Run all debug checks
make debug CYCLE=2

# Or run individual checks
uv run python debug/value_head_check.py --checkpoint checkpoints/models/c1_cycle2_last.pth --buffer checkpoints/buffers/replay_c1_cycle2.pkl
```

## Environment Variables

The Makefile sets `PYTHONPATH` automatically, so all Python commands work correctly.

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'utils.paths'`
- **Solution**: The Makefile sets PYTHONPATH automatically. If you run commands directly, use: `PYTHONPATH=. uv run python ...`

**Issue**: Config not found
- **Solution**: Make sure the config file exists in `configs/` directory. Check with: `ls configs/phaseC_c*.py`

**Issue**: Buffer not found
- **Solution**: Run self-play first: `make self-play CYCLE=2`

