"""Shared test configuration."""

import pytest
from hypothesis import settings

# Deeper fuzzing for CI's dedicated property-test run; invoke with
# `pytest -m hypothesis --hypothesis-profile=thorough`. Dev/default stays fast.
settings.register_profile("thorough", max_examples=2500)

# Default per-test *peak* memory ceiling, applied to every test that doesn't set its
# own. Enforced only under `pytest --memray` (inert on ordinary runs); the "Memory
# limits" CI job runs with that flag. Set well above any legitimate test (the suite
# peaks ~1.3 GB) but far below the ~16 GB CI runner, so a regression like the fused
# per-group bootstrap blowup (~15 GB) fails loudly here instead of OOM-killing CI.
# Override per test with an explicit ``@pytest.mark.limit_memory("N GB")``.
_DEFAULT_MEMORY_LIMIT = "6 GB"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Apply a default memory ceiling and auto-mark Hypothesis tests.

    The memory ceiling (``limit_memory``) is enforced only under ``--memray``; the
    Hypothesis mark lets the property tests be deselected with ``-m 'not hypothesis'``.
    """
    for item in items:
        # Default memory ceiling for any test that hasn't set an explicit override.
        if item.get_closest_marker("limit_memory") is None:
            item.add_marker(pytest.mark.limit_memory(_DEFAULT_MEMORY_LIMIT))
        if any(
            marker.name == "given" or (hasattr(marker, "args") and "hypothesis" in str(marker))
            for marker in item.iter_markers()
        ):
            item.add_marker(pytest.mark.hypothesis)
            continue
        # Also catch tests in *Hypothesis classes or with _property suffix.
        # `cls` only exists on class-collected items, not the base Item type.
        cls = getattr(item, "cls", None)
        if "Hypothesis" in (cls.__name__ if cls else "") or item.name.endswith("_property"):
            item.add_marker(pytest.mark.hypothesis)
