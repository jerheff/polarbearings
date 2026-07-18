"""Root conftest: run the README's Python code blocks as tests via Sybil.

Only ``README.md`` is collected here (the package's user-facing examples); the
main suite under ``tests/`` is untouched, since pytest never walks the repo root
during an ordinary ``pytest tests/`` run. Invoke the doc checks by naming the
file, e.g. ``pytest README.md`` (see ``just doctest`` and the ``doctest`` CI job).

The blocks execute against the real API in document order, sharing one namespace,
so a renamed metric or changed signature fails loudly. The example frames the
prose reuses (``df``/``reg``) are seeded from ``<!--- invisible-code-block --->``
HTML comments, which render as nothing on GitHub but are evaluated by Sybil.
"""

from sybil import Sybil
from sybil.parsers.markdown import PythonCodeBlockParser

pytest_collect_file = Sybil(
    parsers=[PythonCodeBlockParser()],
    patterns=["README.md"],
).pytest()
