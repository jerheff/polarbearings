"""Bump the CI's "recent" Polars leg to the newest stable Polars release on PyPI.

The recent leg is single-sourced in ``.github/workflows/test.yml``'s ``config`` job
(``polars-recent=X.Y.Z``); the justfile reads it from there too, so this rewrites that one
line and nothing else. Run by the scheduled ``bump-polars.yml`` workflow, or locally via
``just bump-polars``. Usage::

    python scripts/bump_polars_recent.py [--dry-run]

"Newest stable" is the highest plain ``x.y.z`` release with at least one non-yanked file
(pre-releases, dev builds, and fully-yanked versions are skipped). Polars is exempt from the
repo's release-age cooldown (see ``[tool.uv.exclude-newer-package]``), so there is no age
gate here either: the bump PR's CI run plus human review is the gate.

Prints ``old -> new`` (or ``up to date``) and, when ``$GITHUB_OUTPUT`` is set, writes
``old``, ``new``, and ``changed`` (``true``/``false``) step outputs. Never downgrades.
"""

import json
import os
import re
import sys
import urllib.request
from pathlib import Path

_PYPI = "https://pypi.org/pypi/polars/json"
_WORKFLOW = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "test.yml"
_PIN = re.compile(r"(polars-recent=)(\d+\.\d+\.\d+)")

Version = tuple[int, int, int]


def _parse(ver: str) -> Version | None:
    """Return ``ver`` as an int triple if it is a plain stable ``x.y.z``, else None."""
    parts = ver.split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        return None
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def latest_stable(releases: dict[str, list[dict[str, object]]]) -> str:
    """Return the highest stable, not-fully-yanked version in a PyPI ``releases`` map.

    Raises:
        SystemExit: If no stable, non-yanked release is present.
    """
    candidates = [
        (pv, ver)
        for ver, files in releases.items()
        if (pv := _parse(ver)) is not None and any(not f.get("yanked") for f in files)
    ]
    if not candidates:
        msg = "no stable Polars release resolved from PyPI"
        raise SystemExit(msg)
    return max(candidates)[1]


def _emit(**outputs: str) -> None:
    """Append ``key=value`` lines to ``$GITHUB_OUTPUT`` when running in Actions."""
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with Path(path).open("a", encoding="utf-8") as fh:
            fh.writelines(f"{k}={v}\n" for k, v in outputs.items())


def main() -> None:
    """Resolve the newest Polars release and rewrite the pin if it is newer.

    Raises:
        SystemExit: If the workflow does not contain exactly one ``polars-recent`` pin.
    """
    dry_run = "--dry-run" in sys.argv[1:]
    text = _WORKFLOW.read_text(encoding="utf-8")
    pins = _PIN.findall(text)
    if len(pins) != 1:
        msg = f"expected exactly one polars-recent=X.Y.Z in {_WORKFLOW}, found {len(pins)}"
        raise SystemExit(msg)
    old = pins[0][1]

    with urllib.request.urlopen(_PYPI, timeout=60) as fh:
        new = latest_stable(json.load(fh)["releases"])

    old_v, new_v = _parse(old), _parse(new)
    if old_v is None or new_v is None or new_v <= old_v:
        print(f"polars-recent {old}: up to date (newest stable on PyPI: {new})")
        _emit(old=old, new=old, changed="false")
        return

    print(f"polars-recent {old} -> {new}")
    if not dry_run:
        _WORKFLOW.write_text(_PIN.sub(rf"\g<1>{new}", text), encoding="utf-8")
    _emit(old=old, new=new, changed="true")


if __name__ == "__main__":
    main()
