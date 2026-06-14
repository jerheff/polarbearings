"""Shared test configuration."""

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Auto-mark Hypothesis tests so they can be deselected with -m 'not hypothesis'."""
    for item in items:
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
