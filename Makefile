# Makefile for AlphaZero Project
# Run `make help` to see available commands

# --- Configuration ---
# Default cycle to run. Override from CLI: make train CYCLE=2
CYCLE ?= 1

# Default cycles for arena comparison. Override from CLI.
CANDIDATE_CYCLE ?= $(CYCLE)
BASELINE_CYCLE ?= $(shell expr $(CYCLE) - 1)

# Paths derived from cycle numbers
CANDIDATE_MODEL := checkpoints/models/c1_cycle$(CANDIDATE_CYCLE)_best.pth
BASELINE_MODEL := checkpoints/models/c1_cycle$(BASELINE_CYCLE)_best.pth
CYCLE_BUFFER := checkpoints/buffers/replay_c1_cycle$(CYCLE).pkl
CYCLE_MODEL_LAST := checkpoints/models/c1_cycle$(CYCLE)_last.pth

# Pass-through arguments for training overrides
# Example: make train CYCLE=2 ARGS="--learning_rate 0.0005"
ARGS ?=

.PHONY: help lint lint-fix format format-all clean all \
        self-play train analyze debug arena

# Set PYTHONPATH to current directory (project root)
export PYTHONPATH := $(shell pwd)

# Show available commands
help:
	@echo "Available commands:"
	@echo "  make self-play [CYCLE=N]"
	@echo "  make train [CYCLE=N] [ARGS=\"--learning_rate 0.001\"]"
	@echo "  make analyze [CYCLE=N]"
	@echo "  make debug [CYCLE=N]"
	@echo "  make arena [CANDIDATE_CYCLE=N] [BASELINE_CYCLE=M]"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint, lint-fix, format, clean, all"

# --- Core Workflow ---
self-play:
	python cli/self_play/self_play_main.py --cycle $(CYCLE) $(ARGS)

train:
	python cli/train/train_loop_main.py --cycle $(CYCLE) $(ARGS)

analyze:
	python cli/self_play/analyze_jsonl.py --cycle $(CYCLE)

debug:
	@echo "Running debug checks for CYCLE=$(CYCLE)"
	python debug/value_head_check.py \
	  --checkpoint $(CYCLE_MODEL_LAST) \
	  --buffer $(CYCLE_BUFFER)
	python debug/policy_head_check.py \
	  --checkpoint $(CYCLE_MODEL_LAST) \
	  --buffer $(CYCLE_BUFFER)
	python debug/training_smoke_check.py \
	  --checkpoint $(CYCLE_MODEL_LAST) \
	  --buffer $(CYCLE_BUFFER)

arena:
	@echo "Comparing CANDIDATE=$(CANDIDATE_MODEL) vs BASELINE=$(BASELINE_MODEL)"
	python scripts/arena.py \
	  --baseline $(BASELINE_MODEL) \
	  --candidate $(CANDIDATE_MODEL) \
	  --games 200 --sims 800 \
	  --out checkpoints/arena/arena_c$(CANDIDATE_CYCLE)_vs_c$(BASELINE_CYCLE).json \
	  --cycle $(CANDIDATE_CYCLE)

# --- Code Quality ---
lint:
	ruff check .

lint-fix:
	ruff check . --fix

format:
	ruff format .
	black .

# --- Cleanup ---
clean:
	@echo "Cleaning checkpoints and logs..."
	@find checkpoints -type f ! -name ".gitkeep" -delete
	@find logs -type f ! -name ".gitkeep" -delete

all: lint-fix format
