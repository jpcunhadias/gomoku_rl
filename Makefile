# Makefile for AlphaZero Project

.PHONY: lint lint-fix format format-black format-all all

# Lint with Ruff (check only, no modifications)
lint:
	ruff check .

# Lint and auto-fix with Ruff
lint-fix:
	ruff check . --fix

# Format with Ruff (e.g., docstring and style)
format:
	ruff format .

# Format with Black (for full PEP8 formatting)
format-black:
	black .

# Apply both Ruff and Black formatters
format-all: format format-black

# Run full cleanup pipeline
all: lint-fix format-all
