VENV_BIN := $(shell if [ -d ".venv/bin" ]; then echo ".venv/bin/"; fi)
RUFF := $(VENV_BIN)ruff
PYTEST := $(VENV_BIN)pytest

.PHONY: lint format test check clean

lint:
	$(RUFF) check .
	$(RUFF) format --check .

format:
	$(RUFF) format .
	$(RUFF) check --fix .

test:
	$(PYTEST) -q

check: lint test

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
