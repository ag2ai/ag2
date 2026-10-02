# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""`ag2[ag-ui]` does not install google-genai, so the AG-UI surface must import without it.

Checked on the source rather than by blocking the import: a module may import
`google` only where an `ImportError` is caught, and everything outside such a
guard would fail at `import ag2.ag_ui` on an install without the extra.
"""

import ast
from pathlib import Path

import pytest

import ag2
from ag2 import ag_ui

_GOOGLE_ROOTS = ("google", "ag2.config.gemini", "ag2.config.vertexai")
_ROOTS = (Path(ag_ui.__file__).parent, Path(ag2.__file__).parent / "a2ui" / "transports")


def _is_google(name: str) -> bool:
    return any(name == root or name.startswith(f"{root}.") for root in _GOOGLE_ROOTS)


def _unguarded_google_imports(node: ast.AST) -> list[int]:
    lines: list[int] = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.Try):
            # the body is guarded; handlers, else and finally are not
            rest = [*child.handlers, *child.orelse, *child.finalbody]
            for stmt in rest:
                lines += _unguarded_google_imports(stmt)
            continue
        if (
            isinstance(child, ast.Import)
            and any(_is_google(a.name) for a in child.names)
            or isinstance(child, ast.ImportFrom)
            and child.level == 0
            and _is_google(child.module or "")
        ):
            lines.append(child.lineno)
        lines += _unguarded_google_imports(child)
    return lines


@pytest.mark.parametrize(
    "path",
    sorted(p for root in _ROOTS for p in root.rglob("*.py")),
    ids=lambda p: p.name,
)
def test_no_module_imports_google_outside_an_import_guard(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))

    assert _unguarded_google_imports(tree) == []
