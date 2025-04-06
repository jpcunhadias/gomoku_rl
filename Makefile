.PHONY: lint lint-fix format

# Run Ruff linter (check only)
lint:
	ruff check .

# Run Ruff linter and automatically fix issues
lint-fix:
	ruff check . --fix

# Apply Ruff formatting (equivalent to black)
format:
	ruff format .
