"""Print the Polars versions for ``just test-sweep``, computed live from PyPI.

Selection (latest patch per minor; stable ``x.y.z`` at or above the support floor):

- every minor released within ``recent_days`` (the last 12 months by default),
- one representative minor per ``older_cadence`` days for everything older, and
- always the support floor.

The list is pulled from PyPI at run time, so it auto-updates as Polars releases —
no hardcoded versions to bump. Usage::

    python scripts/compat_versions.py [recent_days] [older_cadence_days]

Prints the chosen versions space-separated on one line (oldest first).
"""

import datetime as dt
import json
import sys
import urllib.request

# The library's support floor (README/pyproject promise "Polars 1.0.0+").
_FLOOR = (1, 0, 0)
_PYPI = "https://pypi.org/pypi/polars/json"


def _latest_patch_per_minor() -> list[tuple[tuple[int, int, int], str, dt.date]]:
    """Return ``(version_tuple, version_str, release_date)`` for each minor's newest patch."""
    with urllib.request.urlopen(_PYPI, timeout=60) as fh:
        releases = json.load(fh)["releases"]
    by_minor: dict[tuple[int, int], tuple[tuple[int, int, int], str, dt.date]] = {}
    for ver, files in releases.items():
        parts = ver.split(".")
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            continue  # skip pre-releases and non-``x.y.z`` tags
        pv = (int(parts[0]), int(parts[1]), int(parts[2]))
        dates = [f["upload_time_iso_8601"] for f in files if not f.get("yanked")]
        if pv < _FLOOR or not dates:
            continue
        date = dt.date.fromisoformat(min(dates)[:10])
        key = (pv[0], pv[1])
        if key not in by_minor or pv > by_minor[key][0]:
            by_minor[key] = (pv, ver, date)
    return sorted(by_minor.values())


def main() -> None:
    """Resolve and print the sweep's Polars versions."""
    recent_days = int(sys.argv[1]) if len(sys.argv) > 1 else 365
    older_cadence = int(sys.argv[2]) if len(sys.argv) > 2 else 365

    minors = _latest_patch_per_minor()
    if not minors:
        sys.exit("no Polars releases resolved from PyPI")
    cutoff = dt.date.today() - dt.timedelta(days=recent_days)

    # Every minor in the recent window.
    selected: dict[tuple[int, int, int], str] = {
        pv: ver for pv, ver, date in minors if date >= cutoff
    }
    # One representative per `older_cadence`, walking back from the cutoff (newest first).
    last = cutoff
    older = sorted((m for m in minors if m[2] < cutoff), key=lambda m: m[2], reverse=True)
    for pv, ver, date in older:
        if last - date >= dt.timedelta(days=older_cadence):
            selected[pv] = ver
            last = date
    # Always include the support floor.
    floor_pv, floor_ver, _ = minors[0]
    selected[floor_pv] = floor_ver

    print(" ".join(ver for _, ver in sorted(selected.items())))


if __name__ == "__main__":
    main()
