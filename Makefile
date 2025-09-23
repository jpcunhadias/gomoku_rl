# Makefile for AlphaZero Project
# Run `make help` to see available commands

.PHONY: help lint lint-fix format format-ruff format-black format-all clean all \
        self-play train debug analyze_buffer eval reset micro_train

# Set PYTHONPATH to current directory (project root)
export PYTHONPATH := $(shell pwd)

# Show available commands
help:
	@echo "Available commands:"
	@echo "  self-play       - Run self-play script"
	@echo "  train           - Run training loop"
	@echo "  eval            - Evaluate model vs. pure MCTS"
	@echo "  debug           - Run debug script for value inspection"
	@echo "  analyze_buffer  - Analyze buffer contents"
	@echo "  lint            - Run Ruff linter (check only)"
	@echo "  lint-fix        - Run Ruff linter and auto-fix"
	@echo "  format          - Run both Ruff and Black formatters"
	@echo "  format-ruff     - Format code with Ruff"
	@echo "  format-black    - Format code with Black (PEP8)"
	@echo "  format-all      - Alias for 'format'"
	@echo "  clean           - Remove artifacts (checkpoints/logs)"
	@echo "  all             - Run lint-fix and format"

# Core functionality
self-play:
	python cli/self_play/self_play_main.py

train:
	python cli/train/train_loop_main.py

eval:
	python cli/eval/eval.py --checkpoint checkpoints/policy_value_net_best.pth --num_games 20 --board_size 8 --eval_sim 800

debug:
	python debug/value_head_check.py \
  --checkpoint checkpoints/policy_value_net_best.pth \
  --buffer checkpoints/replay_buffer.pkl \
  --batch 256 \
  --output debug/debug_outputs && \
	python debug/policy_head_check.py \
  --checkpoint checkpoints/policy_value_net_best.pth \
  --buffer checkpoints/replay_buffer.pkl \
  --batch 512 && \
	python debug/training_smoke_check.py

analyze_buffer:
	python cli/self_play/analyze_buffer.py

# Code quality
lint:
	ruff check .

lint-fix:
	ruff check . --fix

format-ruff:
	ruff format .

format-black:
	black .

format format-all: format-ruff format-black

# Cleanup
clean:
	@echo "Cleaning checkpoints and logs..."
	@find checkpoints -type f ! -name ".gitkeep" -delete
	@find logs -type f ! -name ".gitkeep" -delete

# Combined quality check
all: lint-fix format

reset:
	python scripts/legacy/reset_value_head.py \
  --ckpt_in checkpoints/policy_value_net_best.pth \
  --ckpt_out checkpoints/policy_value_net_reset_value.pth

micro_train:
	python scripts/legacy/micro_train_value_stabilize_v3.py


