"""Bump dev-dependency floors in ``pyproject.toml`` to their newest resolvable versions.

Discovery is delegated to uv: ``uv pip compile`` resolves every ``[dependency-groups]``
requirement at ``--resolution highest``, which already honors ``[tool.uv] exclude-newer``
(the release-age cooldown) and ``requires-python`` — and does so without touching
``uv.lock``. This script only maps those resolved versions onto the ``>=`` floors and
rewrites them in place, preserving comments and formatting: the one step uv has no
primitive for (``uv add`` leaves an already-satisfied floor untouched). ``just autoupdate``
re-locks afterward.

``polars`` lives in ``[project.dependencies]``, not a dependency group, so it is never
rewritten here: it is the sole runtime dep whose low floor is the package's only
version-support promise. Prints one ``name  old -> new`` line per bump.

Run from the project root (``just autoupdate`` does): ``python scripts/bump_dev_floors.py``.
"""

import re
import subprocess
import tomllib

_PYPROJECT = "pyproject.toml"
# A dependency-group requirement: name, optional [extras], then a ">=" lower bound.
_REQ = re.compile(r"^([A-Za-z0-9_.\-]+)(\[[^\]]*\])?\s*>=\s*([^,;\s]+)")
# A `uv pip compile` output pin: `name==version`, before any `; marker` or `# comment`.
_PIN = re.compile(r"^([A-Za-z0-9_.\-]+)==([^\s;]+)")


def _canonical(name: str) -> str:
    """Normalize a distribution name to its PEP 503 form for matching."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _release(ver: str) -> tuple[int, ...]:
    """Numeric release tuple of a version string, e.g. ``"7.1.0"`` -> ``(7, 1, 0)``."""
    return tuple(int(p) for p in re.findall(r"\d+", ver))


def _newer(candidate: str, current: str) -> bool:
    """True if ``candidate`` is a strictly newer release than ``current`` (zero-padded)."""
    a, b = _release(candidate), _release(current)
    width = max(len(a), len(b))
    return a + (0,) * (width - len(a)) > b + (0,) * (width - len(b))


def _resolved(groups: list[str]) -> dict[str, str]:
    """Resolve every dependency group at highest, returning ``{canonical name: version}``.

    Delegates the version policy wholesale to uv — ``exclude-newer`` and ``requires-python``
    are applied by the resolver — and never writes ``uv.lock`` (``uv pip compile`` prints to
    stdout). A universal/marker-split resolution can name a package twice; the lowest is
    kept, since a ``>=`` floor must hold across the whole supported Python range.
    """
    flags = [f"--group={g}" for g in groups]
    out = subprocess.run(
        [
            "uv",
            "pip",
            "compile",
            _PYPROJECT,
            *flags,
            "--resolution",
            "highest",
            "--no-header",
            "--no-annotate",
            "--quiet",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    resolved: dict[str, str] = {}
    for line in out.splitlines():
        match = _PIN.match(line.strip())
        if not match:
            continue
        name, ver = _canonical(match.group(1)), match.group(2)
        if name not in resolved or _newer(resolved[name], ver):
            resolved[name] = ver
    return resolved


def main() -> None:
    """Rewrite dev-dependency floors in ``pyproject.toml`` to uv's resolved versions."""
    with open(_PYPROJECT, encoding="utf-8") as fh:
        src = fh.read()
    groups = tomllib.loads(src).get("dependency-groups", {})
    resolved = _resolved(list(groups))

    changes: list[tuple[str, str, str]] = []
    for group in groups.values():
        for spec in group:
            if not isinstance(spec, str):
                continue  # skip {include-group = "..."} table entries
            match = _REQ.match(spec)
            if not match:
                continue
            name, extras, old = match.group(1), match.group(2) or "", match.group(3)
            new = resolved.get(_canonical(name))
            if new is None or not _newer(new, old):
                continue
            # Replace only this requirement's version digits, keeping quotes/extras/comment.
            pattern = re.compile(r"([\"'])" + re.escape(name) + r"(\[[^\]]*\])?>=" + re.escape(old))
            src, n = pattern.subn(
                lambda m, nm=name, nv=new: f"{m.group(1)}{nm}{m.group(2) or ''}>={nv}", src
            )
            if n:
                changes.append((f"{name}{extras}", old, new))

    with open(_PYPROJECT, "w", encoding="utf-8") as fh:
        fh.write(src)

    for name, old, new in sorted(changes):
        print(f"   {name:<28} {old} -> {new}")
    if not changes:
        print("   (all dev floors already current)")


if __name__ == "__main__":
    main()
