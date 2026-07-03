set shell := ["bash", "-cu"]

default:
    @just --list

# Sync the environment from uv.lock (project + dev group)
install:
    uv sync

# Run the pytest suite
test:
    uv run pytest tests/ -v

# Run all lint and formatting checks. Fails if any issues are found.
lint:
    uv run ruff check src tests ci
    uv run ruff format --check src tests ci

# Apply the formatter and auto-fix lint issues where possible
format:
    uv run ruff format src tests ci
    uv run ruff check --fix src tests ci

# Static type checking
typecheck:
    uv run mypy

# Run all pre-commit hooks against the whole repo
precommit:
    uv run pre-commit run --all-files --show-diff-on-failure

# Build sdist + wheel into ./dist/
build:
    uv build

# Remove build, cache, and tooling artifacts
clean:
    rm -rf build/ dist/ .mypy_cache/ .pytest_cache/ .ruff_cache/ *.egg-info src/*.egg-info
