# Polarbearings Development Commands
# Install just: https://github.com/casey/just
# Documentation: docs/guides/TESTING.md

# Shared benchmark flags. Median/IQR (robust for skewed microbenchmark
# distributions), warmup on (polars compiles query plans on first call), GC
# disabled. min-rounds=5: the slow large-n cases are big deterministic
# computations with low round-to-round variance, so 5 rounds give the same
# median as 20 with far less time/heat. min-rounds is the binding constraint for
# slow ops (it takes precedence over the 1s default --benchmark-max-time); small-n
# still runs many rounds, capped by max-time, so it stays well-sampled.
bench_flags := "--benchmark-only --benchmark-warmup=on --benchmark-disable-gc " + \
    "--benchmark-min-rounds=5 --benchmark-calibration-precision=10 " + \
    "--benchmark-columns=median,iqr,ops,rounds --benchmark-sort=name --benchmark-group-by=group"

# The "recent" Polars leg used by test-compat / coverage / memray / bench recipes.
# Single source of truth — bump here to move the latest tested version.
polars_recent := "1.43.2"

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

# Execute the doc examples as tests: the docstring examples in src/ (exact Polars
# table output is asserted) plus the README's Python blocks (run via Sybil — see
# the root conftest.py). Version-stable across the supported Polars range, so it
# runs in the plain dev env; mirrors the `doctest` CI job. A renamed arg or a
# changed result fails here.
# `--randomly-dont-reorganize`: the README's Sybil examples share one namespace
# and must run in document order, so pytest-randomly's shuffling is disabled here
# (doctests are order-independent). Unlike `-p no:randomly` this keeps the plugin
# loaded, so the `required_plugins` check still holds.
doctest:
    uv run pytest --randomly-dont-reorganize --doctest-modules src/polarbearings/ README.md -q

# Deep property-based fuzz: the Hypothesis-marked tests at 2500 examples each
# (~3.5 min). For local use — `just ci` already runs these at the default ~100
# examples, so this is the on-demand extended pass, not a CI gate.
test-thorough:
    uv run pytest tests/ -m hypothesis --hypothesis-profile=thorough -q

# Test against a specific Polars version (uses ephemeral overlay, venv unchanged)
test-polars version:
    uv run --with polars=={{version}} pytest tests/ -q --tb=short

# Test against min, ~1 year old, and latest Polars versions
test-compat:
    just test-polars 1.0.0
    just test-polars 1.24.0
    just test-polars {{polars_recent}}

# Thorough LOCAL Polars sweep (not a CI gate): every minor from the last 12 months,
# one per year for older releases, plus the floor — the version list is pulled live
# from PyPI (auto-updates), so no hardcoded versions to bump. Runs the full suite
# against each and prints a pass/fail summary. Densify older coverage by shortening
# the second window, e.g. `just test-sweep 365 180` (recent_days, older_cadence_days).
test-sweep recent_days="365" older_cadence="365":
    #!/usr/bin/env bash
    set -uo pipefail
    versions=$(uv run --quiet python scripts/compat_versions.py {{recent_days}} {{older_cadence}})
    if [[ -z "$versions" ]]; then echo "no versions resolved (PyPI unreachable?)"; exit 1; fi
    echo "Sweeping $(echo "$versions" | wc -w | tr -d ' ') Polars versions: $versions"
    fail=0; summary=""
    for v in $versions; do
        printf '\n========== polars %s ==========\n' "$v"
        if uv run --quiet --with "polars==$v" pytest tests/ -q; then
            summary+="  PASS  $v"$'\n'
        else
            summary+="  FAIL  $v"$'\n'; fail=1
        fi
    done
    printf '\n===== sweep summary =====\n%s' "$summary"
    exit "$fail"

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

# Build the sdist + wheel into dist/ (uv build backend) and validate metadata.
build:
    rm -rf dist
    uv build
    uvx twine check dist/*

# Dry-run publish to Test PyPI (auth: UV_PUBLISH_TOKEN with a test.pypi.org token).
publish-test: build
    uv publish --index testpypi

# Smoke-test the published Test PyPI build in a throwaway env (deps from real PyPI).
install-test:
    uv run --no-project --refresh-package polarbearings \
        --index testpypi --index https://pypi.org/simple/ --index-strategy unsafe-best-match \
        --with polarbearings python -c "import polarbearings; print('OK', len(polarbearings.__all__), 'names')"

# Publish to real PyPI (auth: UV_PUBLISH_TOKEN with a pypi.org token).
publish: build
    uv publish

# Coverage gate (100%, branch). Combine a floor (1.0.0) run with a 1.36+ run so the
# version-gated fast paths (e.g. ECE/MCE over-in-agg >=1.36) execute on both arcs —
# no `# pragma: no cover` needed. Mirrors the CI coverage job; the 1.36+ leg is the
# matrix's latest. `coverage report` enforces fail_under.
test-cov:
    rm -f .coverage .coverage.*
    uv run --with polars==1.0.0 coverage run --parallel-mode -m pytest -m 'not hypothesis' -q
    uv run --with polars=={{polars_recent}} coverage run --parallel-mode -m pytest -m 'not hypothesis' -q
    uv run coverage combine
    uv run coverage report --show-missing

# Enforce per-test memory ceilings (tests/conftest.py) via pytest-memray, pinned to a
# recent Polars where the memory pathologies show up. Mirrors the CI memory-limits job.
test-memory:
    uv sync --group memory
    uv run --with polars=={{polars_recent}} pytest -m 'not hypothesis' --memray -q

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
# A thermal cooldown between the runs (see benchmarks/cooldown.py) keeps the
# second run from starting hot — otherwise throttling, not Polars, inflates it.
# The newer version runs FIRST (guaranteed cold from the fresh baseline) since
# its numbers matter most; the floor runs second after the cooldown recovers.
# compare.py labels floor/latest by parsed version, so run order doesn't matter.
# Run on a cool, idle machine for the cross-version ratio to mean anything.
# Compare Polars performance across versions, attributable to polars alone
bench-compare:
    rm -rf .benchmarks
    uv run python benchmarks/cooldown.py baseline
    just bench-polars {{polars_recent}}
    uv run python benchmarks/cooldown.py wait
    just bench-polars 1.0.0
    uv run pytest-benchmark compare --group-by=name --sort=name --columns=median,iqr
    # Doc-ready Markdown (speedup vs sklearn, version ratios) from the two saved runs.
    uv run python benchmarks/compare.py

# Serve the docs site locally with live reload at http://127.0.0.1:8000
docs-serve:
    uv run --group docs mkdocs serve

# Build the docs site into site/ with --strict (fails on broken links/nav),
# mirroring what Read the Docs builds (see .readthedocs.yaml).
docs-build:
    uv run --group docs mkdocs build --strict

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

# Fast local check (lint + type-check + doctests + tests) — NOT full CI (no
# coverage gate, compat matrix, test-highest, or test-memory; run those for the rest)
check: quality doctest test

# Apply all available maintenance upgrades in one shot: refresh prek hooks (prek.toml),
# re-pin GitHub Actions to their latest release (pinact, .github/workflows/), and raise
# every dev-dependency floor to its newest release (pyproject.toml) — then re-lock with a
# transitive sweep (uv.lock). Every step honors a release-age cooldown so nothing just
# published is adopted, each configured in its own tool: prek's `update.cooldown_days`,
# pinact's `min_age`, and uv's `[tool.uv] exclude-newer` (which also gates the floors —
# see below). polars is the SOLE runtime dep and is deliberately left at its low floor
# (the only version-support promise the package makes); nothing else is held low. This
# MUTATES the tree — review `git diff`, then run `just check` && `just docs-build` first.
autoupdate:
    #!/usr/bin/env bash
    set -uo pipefail
    echo "── prek hooks (prek.toml) ────────────────────────────────"
    # freeze + cooldown_days come from the [update] table in prek.toml.
    prek autoupdate || echo "   (prek autoupdate failed; skipped)"
    echo
    echo "── GitHub Actions (pinact, .github/workflows) ────────────"
    # pinact re-pins every action to its latest release SHA; min_age (in .pinact.yaml)
    # applies the cooldown. Needs a token for the GitHub API (release dates + SHAs).
    if command -v pinact >/dev/null; then
        GITHUB_TOKEN="$(gh auth token)" pinact run --update || echo "   (pinact failed; skipped)"
    else
        echo "   (pinact not installed — run 'brew install pinact'; skipped)"
    fi
    echo
    echo "── dev dependency floors (pyproject.toml) ────────────────"
    # The floor script resolves each dep at highest via `uv pip compile` (which honors
    # `[tool.uv] exclude-newer` + requires-python, and does NOT touch uv.lock) and rewrites
    # the `>=` floors to match — so every floor stays inside the cooldown the re-lock then
    # enforces. uv owns the version policy; the script only does the in-place TOML edit.
    uv run --quiet --frozen --no-sync python scripts/bump_dev_floors.py
    uv lock --quiet --upgrade \
      || { echo "   re-lock FAILED — new floors may not resolve together; revert pyproject.toml"; exit 1; }
    uv sync --quiet
    echo
    echo "✓ autoupdate complete — review 'git diff', then: just check && just docs-build"

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
