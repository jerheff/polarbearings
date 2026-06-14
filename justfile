# Polarbear Development Commands
# Install just: https://github.com/casey/just
# Documentation: docs/guides/TESTING.md

# List all available commands
default:
    @just --list

# Install dependencies and pre-commit hooks
setup:
    uv sync
    prek install

# Verify repo is fully operational
health: quality test

# Run all tests with verbose output
test:
    uv run pytest tests/ -v

# Run tests in quiet mode (faster)
test-fast:
    uv run pytest tests/ -q

# Test against a specific Polars version (uses ephemeral overlay, venv unchanged)
test-polars version:
    uv run --with polars=={{version}} pytest tests/ -q --tb=short

# Test against min, ~1 year old, and latest Polars versions
test-compat:
    just test-polars 1.0.0
    just test-polars 1.24.0
    just test-polars 1.38.1

# Test the UPPER bound: newest compatible deps (the dev default is the floor).
# Mirrors the test-highest CI job; leaves the committed lock untouched.
test-highest:
    uv run --isolated --resolution highest pytest tests/ -q --tb=short

# Run tests with coverage report
test-cov:
    uv run pytest --cov=src/polarbear --cov-report=term-missing tests/

# Run performance benchmarks
bench:
    uv run pytest benchmarks/ -v --benchmark-only --benchmark-min-rounds=3 --benchmark-group-by=group --benchmark-columns=mean,stddev

# Check code style with ruff
lint:
    uv run ruff check src/ tests/

# Fix code style issues automatically
lint-fix:
    uv run ruff check --fix src/ tests/

# Format code with ruff
format:
    uv run ruff format src/ tests/

# Run type checking with ty (whole project: src, tests, benchmarks)
type-check:
    uv run ty check

# Run all quality checks (lint + type-check)
quality: lint type-check

# Run mutation testing
mutant:
    rm -rf mutants/
    uv run mutmut run

# Run all CI checks locally (quality + tests)
ci: quality test

# Clean up cache files and artifacts
clean:
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".hypothesis" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete
    find . -type f -name ".coverage" -delete
    rm -rf mutants/
    @echo "✓ Cleaned up cache files"

# Full development setup: install dependencies and run checks
dev: setup quality test
    @echo "✓ Development environment ready!"

# Quick pre-commit check
pre-commit: lint-fix format test-fast
    @echo "✓ Pre-commit checks passed!"
