"""Thermal cooldown helper for benchmarking — no sudo or temperature sensors.

CPU throttling makes a fixed workload run slower, so an **all-core** canary is a
direct proxy for "has the machine recovered": record a cool baseline before any
load, then ``wait`` until the canary returns to within tolerance of that baseline
(or a max-wait cap is hit).

The canary must load *every* core. A single-core proxy is useless: once the
all-core Polars benchmark stops, one core instantly hits single-core turbo (the
rest are idle), so it reads near-baseline while the package is hot and would
throttle under the next all-core run. (A numpy/BLAS matmul barely threads on this
hardware, so it's out too.) We use a Polars sort — the benchmark's own
multi-threaded engine, which scales ~5x across cores — so the canary throttles
exactly like the benchmark does.

This matters because ``just bench-compare`` runs two Polars versions
sequentially; without a real cooldown the second starts hot and throttling (not
Polars) inflates it — which is exactly what flips the version-ratio sign when you
swap the run order.

Usage:
    python benchmarks/cooldown.py baseline   # run first, while the machine is cool
    python benchmarks/cooldown.py wait        # block until recovered to baseline
"""

import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

_REF = Path(".benchmarks/.cool_baseline")
# Fixed all-core workload: a Polars sort uses every core (the same engine the
# benchmark stresses), so its runtime tracks the all-core clock and reflects the
# throttle the next run would hit.
_DF = pl.DataFrame({"x": np.random.default_rng(0).random(8_000_000)})


def _canary() -> float:
    """Time a fixed all-core Polars workload (lower = cooler / higher all-core clock)."""
    start = time.perf_counter()
    for _ in range(8):
        _DF.select(pl.col("x").sort())
    return time.perf_counter() - start


def baseline() -> None:
    """Record the cool-machine canary time (best of a few) for ``wait`` to target."""
    best = min(_canary() for _ in range(5))
    _REF.parent.mkdir(parents=True, exist_ok=True)
    _REF.write_text(str(best))
    print(f"cool baseline: {best * 1000:.0f} ms all-core canary")


def wait(tol: float = 0.05, step: int = 15, max_wait: int = 300, min_sleep: int = 60) -> None:
    """Block until the canary recovers AND a minimum cooldown has elapsed.

    The memory-bound sort canary under-reports the throttle that compute-heavy
    metrics suffer (it recovers ~10% while they swing ~2x), so requiring at least
    ``min_sleep`` seconds guarantees real cooling even when the canary says it's
    fine. Returns early at ``max_wait`` to bound the worst case.
    """
    base = float(_REF.read_text()) if _REF.exists() else min(_canary() for _ in range(3))
    waited = 0
    while True:
        ratio = min(_canary() for _ in range(2)) / base
        if ratio <= 1 + tol and waited >= min_sleep:
            print(f"recovered: canary {ratio:.2f}x baseline after {waited}s")
            return
        if waited >= max_wait:
            print(f"cooldown cap {max_wait}s hit (canary {ratio:.2f}x baseline)")
            return
        print(f"  canary {ratio:.2f}x baseline; waited {waited}s; sleeping {step}s")
        time.sleep(step)
        waited += step


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "wait"
    {"baseline": baseline, "wait": wait}[mode]()
