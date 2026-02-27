# Polarbear Development Commands
# Install just: https://github.com/casey/just
# Documentation: docs/guides/TESTING.md

# List all available commands
default:
    @just --list

# Install dependencies
install:
    uv sync

# Run all tests with verbose output
test:
    uv run pytest tests/ -v

# Run tests in quiet mode (faster)
test-fast:
    uv run pytest tests/ -q

# Test against minimum and latest Polars versions
test-versions:
    uv run python test_versions.py --min-max --no-benchmark

# Test against all supported Polars versions
test-versions-all:
    uv run python test_versions.py

# Run tests with coverage report
test-cov:
    uv run pytest --cov=src/polarbear --cov-report=term-missing tests/

# Run performance benchmarks
bench:
    uv run python benchmark.py

# Check code style with ruff
lint:
    uv run ruff check src/ tests/

# Fix code style issues automatically
lint-fix:
    uv run ruff check --fix src/ tests/

# Format code with ruff
format:
    uv run ruff format src/ tests/

# Check if code is formatted correctly
format-check:
    uv run ruff format --check src/ tests/

# Run type checking with pyright
type-check:
    uv run pyright src/polarbear

# Run all quality checks (lint + type-check)
quality: lint type-check

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
    @echo "✓ Cleaned up cache files"

# Full development setup: install dependencies and run checks
dev: install quality test
    @echo "✓ Development environment ready!"

# Quick pre-commit check
pre-commit: lint-fix format test-fast
    @echo "✓ Pre-commit checks passed!"
