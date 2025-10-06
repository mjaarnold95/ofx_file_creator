SHELL := /bin/zsh
.PHONY: venv typecheck test clean

# Create a Python 3.11 virtual environment in .venv and install runtime + dev deps
venv:
	@if [ ! -d .venv ]; then \
		python3.11 -m venv .venv && \
		.venv/bin/python -m pip install --upgrade pip; \
	fi
	.venv/bin/python -m pip install -r requirements.txt
	.venv/bin/python -m pip install --upgrade mypy pandas-stubs pytest

# Run static type checks (expects mypy.ini at repo root)
typecheck: venv
	.venv/bin/python -m mypy --config-file mypy.ini utils

# Run the test suite
test: venv
	.venv/bin/python -m pytest -q

# Remove generated virtualenv and python caches
clean:
	@echo "Removing .venv and __pycache__ directories..."
	-@rm -rf .venv
	-@find . -type d -name "__pycache__" -prune -exec rm -rf {} +
