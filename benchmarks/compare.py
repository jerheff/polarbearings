"""Generate doc-ready Markdown comparing two pytest-benchmark runs.

Reads two ``pytest-benchmark`` JSON result files — a *floor* Polars run and a
*latest* Polars run, each containing both ``polarbear`` and ``sklearn``
benchmarks — and emits three Markdown sections:

* **speedup vs sklearn** — ``sklearn_median / polarbear_median`` (higher means
  polarbear is faster), one column per Polars version;
* **Polars version ratio** — ``latest_median / floor_median`` for polarbear
  (``> 1`` means the newer Polars is slower);
* **polarbear-only medians** — absolute median time for metrics that have no
  sklearn baseline (e.g. ``gini``).

Each section is wrapped in ``<!-- BEGIN:... -->`` / ``<!-- END:... -->`` markers
so it can be embedded into a docs file and refreshed in place with ``--write``.

The computation is intentionally fixed and deterministic so numbers quoted in
the documentation are always produced the same way:

    speedup(version)      = median(test_sklearn_<m>[n]) / median(test_polarbear_<m>[n])
    version_ratio         = median(latest test_polarbear_<m>[n]) / median(floor ...)

Usage::

    python benchmarks/compare.py [FLOOR.json] [LATEST.json]
    python benchmarks/compare.py --write docs/technical/PERFORMANCE.md
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

# Section ids -> the marker names used in the Markdown / docs.
SPEEDUP = "speedup-vs-sklearn"
VERSION_RATIO = "polars-version-ratio"
PB_ONLY = "polarbear-only-medians"

_NAME_RE = re.compile(r"^test_(polarbear|sklearn)_(.+?)\[(\d+)\]$")
_VERSION_RE = re.compile(r"polars_(\d+)_(\d+)_(\d+)")


def load_medians(path: Path) -> dict[str, float]:
    """Map each benchmark name to its median time (seconds)."""
    data = json.loads(path.read_text())
    return {b["name"]: float(b["stats"]["median"]) for b in data["benchmarks"]}


def version_label(path: Path) -> str:
    """Derive a human label like ``polars 1.41.2`` from the saved file name."""
    m = _VERSION_RE.search(path.stem)
    if m:
        return f"polars {m.group(1)}.{m.group(2)}.{m.group(3)}"
    return path.stem


def parse_name(name: str) -> tuple[str, str, int] | None:
    """Split ``test_<kind>_<metric>[<n>]`` into ``(kind, metric, n)``."""
    m = _NAME_RE.match(name)
    if not m:
        return None
    return m.group(1), m.group(2), int(m.group(3))


def _metrics(medians: dict[str, float]) -> dict[tuple[str, int], dict[str, float]]:
    """Index medians by ``(metric, n)`` -> ``{"polarbear": .., "sklearn": ..}``."""
    out: dict[tuple[str, int], dict[str, float]] = {}
    for name, med in medians.items():
        parsed = parse_name(name)
        if parsed is None:
            continue
        kind, metric, n = parsed
        out.setdefault((metric, n), {})[kind] = med
    return out


def _fmt_n(n: int) -> str:
    return f"{n:,}"


def _section(marker: str, caption: str, header: list[str], rows: list[list[str]]) -> str:
    """Render a marker-wrapped Markdown table."""
    lines = [f"<!-- BEGIN:{marker} -->", f"_{caption}_", ""]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    lines.append(f"<!-- END:{marker} -->")
    return "\n".join(lines)


def render_speedup(
    floor: dict[tuple[str, int], dict[str, float]],
    latest: dict[tuple[str, int], dict[str, float]],
    floor_label: str,
    latest_label: str,
) -> str:
    """Section A: speedup vs sklearn, one column per Polars version."""
    keys = sorted(k for k in floor if "sklearn" in floor[k] and "polarbear" in floor[k])
    rows: list[list[str]] = []
    for metric, n in keys:
        f = floor[(metric, n)]
        f_sp = f"{f['sklearn'] / f['polarbear']:.1f}x"
        lat = latest.get((metric, n), {})
        l_sp = (
            f"{lat['sklearn'] / lat['polarbear']:.1f}x"
            if "sklearn" in lat and "polarbear" in lat
            else "—"
        )
        rows.append([metric, _fmt_n(n), f_sp, l_sp])
    caption = (
        "Speedup vs scikit-learn = sklearn median ÷ polarbear median (higher = polarbear faster)."
    )
    return _section(SPEEDUP, caption, ["Metric", "n", floor_label, latest_label], rows)


def render_version_ratio(
    floor: dict[tuple[str, int], dict[str, float]],
    latest: dict[tuple[str, int], dict[str, float]],
    floor_label: str,
    latest_label: str,
) -> str:
    """Section B: polarbear floor-vs-latest median ratio."""
    keys = sorted(k for k in floor if "polarbear" in floor[k] and "polarbear" in latest.get(k, {}))
    rows: list[list[str]] = []
    for metric, n in keys:
        ratio = latest[(metric, n)]["polarbear"] / floor[(metric, n)]["polarbear"]
        rows.append([metric, _fmt_n(n), f"{ratio:.2f}"])
    caption = (
        f"polarbear median on {latest_label} ÷ {floor_label} "
        "(> 1.00 = newer Polars slower; numpy/sklearn held fixed)."
    )
    return _section(VERSION_RATIO, caption, ["Metric", "n", "ratio"], rows)


def render_pb_only(
    floor: dict[tuple[str, int], dict[str, float]],
    latest: dict[tuple[str, int], dict[str, float]],
    floor_label: str,
    latest_label: str,
) -> str:
    """Section C: absolute medians for metrics with no sklearn baseline."""
    keys = sorted(k for k in floor if "sklearn" not in floor[k] and "polarbear" in floor[k])
    rows: list[list[str]] = []
    for metric, n in keys:
        f_ms = f"{floor[(metric, n)]['polarbear'] * 1e3:.3f}"
        lat = latest.get((metric, n), {})
        l_ms = f"{lat['polarbear'] * 1e3:.3f}" if "polarbear" in lat else "—"
        rows.append([metric, _fmt_n(n), f_ms, l_ms])
    caption = "polarbear-only metrics (no sklearn baseline): absolute median time in ms."
    return _section(
        PB_ONLY, caption, ["Metric", "n", f"{floor_label} (ms)", f"{latest_label} (ms)"], rows
    )


def build_snippets(floor_path: Path, latest_path: Path) -> dict[str, str]:
    """Produce the three Markdown sections keyed by marker id."""
    floor = _metrics(load_medians(floor_path))
    latest = _metrics(load_medians(latest_path))
    fl, ll = version_label(floor_path), version_label(latest_path)
    return {
        SPEEDUP: render_speedup(floor, latest, fl, ll),
        VERSION_RATIO: render_version_ratio(floor, latest, fl, ll),
        PB_ONLY: render_pb_only(floor, latest, fl, ll),
    }


def default_runs() -> tuple[Path, Path]:
    """Discover the floor/latest run JSONs saved by ``just bench-compare``."""
    found = sorted(glob.glob(".benchmarks/**/0*_polars_*.json", recursive=True))
    if len(found) < 2:
        sys.exit(
            "error: could not find two saved runs under .benchmarks/. "
            "Run `just bench-compare` first, or pass FLOOR and LATEST paths."
        )
    # Saved in order: floor (lowest NNNN prefix) then latest.
    return Path(found[0]), Path(found[-1])


def write_into(doc: Path, snippets: dict[str, str]) -> None:
    """Replace marker-delimited regions in ``doc`` with freshly rendered tables."""
    text = doc.read_text()
    replaced = 0
    for marker, snippet in snippets.items():
        begin, end = f"<!-- BEGIN:{marker} -->", f"<!-- END:{marker} -->"
        b_idx, e_idx = text.find(begin), text.find(end)
        if b_idx == -1 and e_idx == -1:
            continue
        if (b_idx == -1) != (e_idx == -1):
            present = begin if b_idx != -1 else end
            sys.exit(f"error: {doc}: marker '{marker}' has a {present} without its matching pair.")
        # ``snippet`` already includes its own BEGIN/END markers; splice it in.
        text = text[:b_idx] + snippet + text[e_idx + len(end) :]
        replaced += 1
    if replaced == 0:
        sys.exit(
            f"error: {doc} contains none of the expected markers "
            f"({', '.join(snippets)}); nothing to write."
        )
    doc.write_text(text)
    print(f"updated {replaced} section(s) in {doc}", file=sys.stderr)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("floor", nargs="?", type=Path, help="floor Polars run JSON")
    parser.add_argument("latest", nargs="?", type=Path, help="latest Polars run JSON")
    parser.add_argument(
        "--write",
        type=Path,
        metavar="MARKDOWN",
        help="embed the snippets into MARKDOWN in place (between BEGIN/END markers)",
    )
    args = parser.parse_args()

    if args.floor and args.latest:
        floor_path, latest_path = args.floor, args.latest
    elif args.floor or args.latest:
        parser.error("provide BOTH floor and latest paths, or neither (to auto-discover)")
    else:
        floor_path, latest_path = default_runs()

    snippets = build_snippets(floor_path, latest_path)

    if args.write:
        write_into(args.write, snippets)
    else:
        print("\n\n".join(snippets[m] for m in (SPEEDUP, VERSION_RATIO, PB_ONLY)))


if __name__ == "__main__":
    main()
