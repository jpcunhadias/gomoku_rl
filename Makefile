# Makefile for AlphaZero Project
# Run `make help` to see available commands

.PHONY: help lint lint-fix format format-ruff format-black format-all clean all self-play train

# Set PYTHONPATH to current directory (project root)
export PYTHONPATH := $(shell pwd)

# Show available commands
help:
	@echo "Available commands:"
	@echo " self-play    - Run self-play script"
	@echo " train         - Run training loop"
	@echo "  lint           - Run Ruff linter (check only)"
	@echo "  lint-fix       - Run Ruff linter and auto-fix"
	@echo "  format         - Run both Ruff and Black formatters"
	@echo "  format-ruff    - Format code with Ruff"
	@echo "  format-black   - Format code with Black (PEP8)"
	@echo "  format-all     - Alias for 'format'"
	@echo "  clean          - Placeholder for cleaning build artifacts"
	@echo "  all            - Run lint-fix and format"

self-play:
	python cli/self_play/self_play_main.py

train:
	python cli/train/train_loop_main.py

# Lint with Ruff (check only)
lint:
	ruff check .

# Lint with Ruff and fix issues
lint-fix:
	ruff check . --fix

# Format with Ruff
format-ruff:
	ruff format .

# Format with Black
format-black:
	black .

# Run both Ruff and Black formatters
format format-all: format-ruff format-black

# Clean up artifacts (optional - update if needed)
clean:
	@echo "Cleaning checkpoints and logs..."
	@find checkpoints -type f ! -name ".gitkeep" -delete
	@find logs -type f ! -name ".gitkeep" -delete

# Run full code quality pipeline
all: lint-fix format
