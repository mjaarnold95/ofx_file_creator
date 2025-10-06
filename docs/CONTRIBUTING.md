# Contributing

Thanks for wanting to contribute! A few small helpers are provided to make
local development and checks consistent across contributors.

## Quick dev setup (Makefile)

The repository includes a small `Makefile` with convenient targets. From the
project root run:

# create and bootstrap the Python 3.11 virtualenv and install dev deps
make venv

# run mypy (uses .venv)
make typecheck

# run tests (uses .venv)
make test

# clean up
make clean

Notes
- The `typecheck` target runs `mypy --config-file mypy.ini utils` inside `.venv`.
- If you don't have Python 3.11 available locally, consider installing it with
  `pyenv` or adjusting the Makefile to use a Python executable you do have.
