# Polarbear Development Commands
# Install just: https://github.com/casey/just
# Documentation: docs/guides/TESTING.md

# Shared benchmark flags. Median/IQR (robust for skewed microbenchmark
# distributions), warmup on (polars compiles query plans on first call),
# GC disabled, and >=20 rounds for stable medians.
bench_flags := "--benchmark-only --benchmark-warmup=on --benchmark-disable-gc " + \
    "--benchmark-min-rounds=20 --benchmark-calibration-precision=10 " + \
    "--benchmark-columns=median,iqr,ops,rounds --benchmark-sort=name --benchmark-group-by=group"

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
    just test-polars 1.41.2

# Test the UPPER bound: newest compatible deps (the dev default is the floor).
# Mirrors the test-highest CI job; leaves the committed lock untouched.
test-highest:
    uv run --isolated --resolution highest pytest tests/ -q --tb=short

# Test the PUBLISHED wheel the way a real user installs it: a clean venv with
# only the dev/test tooling, then the wheel — which must pull its OWN runtime
# deps (polars). Catches missing/incorrect dependency declarations and packaging
# bugs the editable dev install masks. Uses no locked versions on purpose.
test-wheel:
    #!/usr/bin/env bash
    set -euo pipefail
    rm -rf dist .venv-wheel
    uv build
    uv venv .venv-wheel
    uv pip install --python .venv-wheel --group dev
    uv pip install --python .venv-wheel dist/*.whl
    uvx twine check dist/*
    .venv-wheel/bin/python -m pytest tests/ -q

# Run tests with coverage report
test-cov:
    uv run pytest --cov=src/polarbear --cov-report=term-missing tests/

# Cap the size sweep with BENCH_MAX_N (e.g. `BENCH_MAX_N=100000 just bench`).
# Run performance benchmarks against the current/dev env
bench:
    uv run pytest benchmarks/ {{ bench_flags }}

# Holds numpy/scikit-learn FIXED (unlike `--resolution highest`), so the result
# is a true ceteris-paribus comparison. Saves the run under .benchmarks/.
# Benchmark against a specific Polars version
bench-polars version:
    uv run --with polars=={{version}} pytest benchmarks/ {{ bench_flags }} \
        --benchmark-save=polars_{{ replace(version, ".", "_") }}

# Runs floor vs latest with ONLY polars changing, then diffs the two runs.
# Compare Polars performance across versions, attributable to polars alone
bench-compare:
    rm -rf .benchmarks
    just bench-polars 1.0.0
    just bench-polars 1.41.2
    uv run pytest-benchmark compare --group-by=name --sort=name --columns=median,iqr
    # Doc-ready Markdown (speedup vs sklearn, version ratios) from the two saved runs.
    uv run python benchmarks/compare.py

# Check code style with ruff (whole project)
lint:
    uv run ruff check

# Fix code style issues automatically
lint-fix:
    uv run ruff check --fix

# Format code with ruff
format:
    uv run ruff format

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

# Report available updates: pyproject deps, prek hooks, and pinned GitHub Actions
outdated:
    #!/usr/bin/env bash
    set -uo pipefail
    echo "── pyproject.toml dependencies (vs current venv) ─────────"
    echo "   note: polars/numpy/scikit-learn floors are pinned LOW on purpose"
    uv pip list --outdated || true
    echo
    echo "── prek hooks (prek.toml) ────────────────────────────────"
    prek autoupdate --dry-run || true
    echo
    echo "── GitHub Actions (.github/workflows) ────────────────────"
    grep -rhoE 'uses: [^ ]+@[a-f0-9]{40} # \S+' .github/workflows/ \
      | sed 's/.*uses: //' | sort -u \
      | while read -r ref _ pinned; do
          repo="${ref%@*}"
          latest=$(gh api "repos/$repo/releases/latest" --jq .tag_name 2>/dev/null || echo "?")
          if [ "$latest" = "$pinned" ] || [ "$latest" = "?" ]; then
            printf '   %-26s %s\n' "$repo" "$pinned"
          else
            printf '   %-26s %s → %s\n' "$repo" "$pinned" "$latest"
          fi
        done

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
