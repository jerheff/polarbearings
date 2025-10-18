#!/usr/bin/env python3
"""Test polarbear against multiple Polars versions locally.

This script makes it easy to test compatibility with different Polars versions
without manually installing and testing each one.

Usage:
    # Test all supported versions
    uv run python test_versions.py

    # Test specific versions
    uv run python test_versions.py --versions 1.0.0 1.34.0

    # Test only minimum and latest
    uv run python test_versions.py --min-max

    # Run with verbose pytest output
    uv run python test_versions.py --verbose

    # Skip benchmark tests for faster execution
    uv run python test_versions.py --no-benchmark
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Minimum supported version
MIN_VERSION = "1.0.0"

# Versions to test by default
DEFAULT_VERSIONS = ["1.0.0", "1.10.0", "1.20.0", "1.34.0"]


def run_command(cmd: list[str], description: str) -> tuple[bool, str]:
    """Run a shell command and return success status and output."""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    output = result.stdout + result.stderr
    print(output)

    return result.returncode == 0, output


def get_current_polars_version() -> str:
    """Get the currently installed Polars version."""
    result = subprocess.run(
        ["uv", "pip", "list"],
        capture_output=True,
        text=True,
    )

    for line in result.stdout.split("\n"):
        if line.startswith("polars "):
            parts = line.split()
            if len(parts) >= 2:
                return parts[1]

    return "unknown"


def install_polars_version(version: str) -> bool:
    """Install a specific Polars version."""
    success, _ = run_command(
        ["uv", "pip", "install", f"polars=={version}"],
        f"Installing Polars {version}"
    )
    return success


def run_tests(verbose: bool = False, skip_benchmark: bool = False) -> bool:
    """Run the test suite."""
    pytest_args = ["uv", "run", "pytest", "tests/"]

    if verbose:
        pytest_args.append("-v")
    else:
        pytest_args.append("-q")

    pytest_args.append("--tb=short")

    success, _ = run_command(pytest_args, "Running tests")

    if not success:
        return False

    # Run benchmark if not skipped
    if not skip_benchmark:
        success, _ = run_command(
            ["uv", "run", "python", "benchmark.py"],
            "Running benchmarks"
        )

    return success


def main():
    parser = argparse.ArgumentParser(
        description="Test polarbear against multiple Polars versions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all default versions
  %(prog)s

  # Test only minimum and latest
  %(prog)s --min-max

  # Test specific versions
  %(prog)s --versions 1.0.0 1.20.0 1.34.0

  # Verbose output
  %(prog)s --verbose

  # Skip benchmarks for faster testing
  %(prog)s --no-benchmark
        """
    )

    parser.add_argument(
        "--versions",
        nargs="+",
        help=f"Polars versions to test (default: {', '.join(DEFAULT_VERSIONS)})"
    )

    parser.add_argument(
        "--min-max",
        action="store_true",
        help=f"Test only minimum ({MIN_VERSION}) and latest versions"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Run pytest with verbose output"
    )

    parser.add_argument(
        "--no-benchmark",
        action="store_true",
        help="Skip benchmark tests for faster execution"
    )

    args = parser.parse_args()

    # Determine which versions to test
    if args.min_max:
        versions_to_test = [MIN_VERSION, DEFAULT_VERSIONS[-1]]
    elif args.versions:
        versions_to_test = args.versions
    else:
        versions_to_test = DEFAULT_VERSIONS

    print("="*80)
    print("Polarbear Version Compatibility Testing")
    print("="*80)
    print(f"Testing against Polars versions: {', '.join(versions_to_test)}")
    print(f"Minimum supported version: {MIN_VERSION}")
    print(f"Verbose output: {args.verbose}")
    print(f"Run benchmarks: {not args.no_benchmark}")
    print("="*80)

    # Save current version to restore later
    original_version = get_current_polars_version()
    print(f"\nCurrent Polars version: {original_version}")

    results = {}

    for version in versions_to_test:
        print(f"\n{'#'*80}")
        print(f"# Testing with Polars {version}")
        print(f"{'#'*80}")

        # Install the version
        if not install_polars_version(version):
            print(f"❌ Failed to install Polars {version}")
            results[version] = False
            continue

        # Run tests
        success = run_tests(verbose=args.verbose, skip_benchmark=args.no_benchmark)
        results[version] = success

        if success:
            print(f"\n✅ Polars {version}: All tests passed!")
        else:
            print(f"\n❌ Polars {version}: Tests failed!")

    # Restore original version
    print(f"\n{'='*80}")
    print(f"Restoring original Polars version: {original_version}")
    print(f"{'='*80}")
    install_polars_version(original_version)

    # Print summary
    print(f"\n{'='*80}")
    print("Test Summary")
    print(f"{'='*80}")

    all_passed = True
    for version, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"Polars {version:10s}: {status}")
        if not success:
            all_passed = False

    print(f"{'='*80}")

    if all_passed:
        print("🎉 All versions passed!")
        return 0
    else:
        print("⚠️  Some versions failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
