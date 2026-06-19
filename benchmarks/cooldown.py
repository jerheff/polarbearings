"""Thermal cooldown helper for benchmarking — no sudo or temperature sensors.

CPU throttling makes a fixed single-core workload run slower, so a tiny CPU
"canary" is a direct proxy for "has the machine recovered": record a cool
baseline before any load, then ``wait`` until the canary returns to within
tolerance of that baseline (or a max-wait cap is hit).

This matters because ``just bench-compare`` runs two Polars versions
sequentially; without a cooldown between them, the second run starts hot and
its times are inflated by throttling, not by Polars.

Usage:
    python benchmarks/cooldown.py baseline   # run first, while the machine is cool
    python benchmarks/cooldown.py wait        # block until recovered to baseline
"""

import sys
import time
from pathlib import Path

import numpy as np

_REF = Path(".benchmarks/.cool_baseline")
# Fixed single-threaded, transcendental (no BLAS threads) CPU work — its runtime
# tracks core clock, so it slows under throttling and recovers as the chip cools.
_A = np.random.default_rng(0).random(8_000_000)


def _canary() -> float:
    """Time a fixed single-core CPU workload (lower = cooler/faster clock)."""
    start = time.perf_counter()
    total = 0.0
    for _ in range(8):
        total += float((np.sin(_A) * np.cos(_A)).sum())
    return time.perf_counter() - start


def baseline() -> None:
    """Record the cool-machine canary time (best of a few) for ``wait`` to target."""
    best = min(_canary() for _ in range(5))
    _REF.parent.mkdir(parents=True, exist_ok=True)
    _REF.write_text(str(best))
    print(f"cool baseline: {best * 1000:.0f} ms canary")


def wait(tol: float = 0.05, step: int = 15, max_wait: int = 300) -> None:
    """Block until the canary is within ``tol`` of the baseline, or ``max_wait`` s."""
    base = float(_REF.read_text()) if _REF.exists() else min(_canary() for _ in range(3))
    waited = 0
    while True:
        ratio = min(_canary() for _ in range(2)) / base
        if ratio <= 1 + tol:
            print(f"recovered: canary {ratio:.2f}x baseline after {waited}s")
            return
        if waited >= max_wait:
            print(f"cooldown cap {max_wait}s hit (canary still {ratio:.2f}x baseline)")
            return
        print(f"  hot: canary {ratio:.2f}x baseline; sleeping {step}s")
        time.sleep(step)
        waited += step


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "wait"
    {"baseline": baseline, "wait": wait}[mode]()
