#!/usr/bin/env python3
"""Generate a pytest test-file scaffold for an autogen.beta module.

Usage::

    python scaffold.py autogen.beta.tools.toolkits.memory
    python scaffold.py autogen.beta.exceptions --out test/beta/test_exceptions.py

The script imports the target module, collects its public symbols (classes and
functions listed in ``__all__`` or without a leading underscore), and emits a
starter test file with one empty test stub per symbol.

The output is printed to stdout by default so you can review it before writing
it anywhere.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import sys
from pathlib import Path
from textwrap import dedent

_HEADER = dedent("""\
    # Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
    #
    # SPDX-License-Identifier: Apache-2.0

    import pytest

    # ---------------------------------------------------------------------------
    # Auto-generated test scaffold — fill in each test body.
    # Run:  pytest {test_path}
    # ---------------------------------------------------------------------------
""")

_ASYNC_STUB = dedent("""\
    @pytest.mark.asyncio
    async def test_{name}() -> None:
        # TODO: test {symbol}
        ...

""")

_SYNC_STUB = dedent("""\
    def test_{name}() -> None:
        # TODO: test {symbol}
        ...

""")


def _snake(name: str) -> str:
    """CamelCase → snake_case."""
    import re

    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    s = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", s)
    return s.lower()


def _is_async(obj: object) -> bool:
    return inspect.iscoroutinefunction(obj) or (
        isinstance(obj, type) and any(inspect.iscoroutinefunction(m) for _, m in inspect.getmembers(obj))
    )


def collect_symbols(module_name: str) -> list[tuple[str, object]]:
    mod = importlib.import_module(module_name)
    names = list(mod.__all__) if hasattr(mod, "__all__") else [n for n in dir(mod) if not n.startswith("_")]

    symbols = []
    for name in names:
        obj = getattr(mod, name, None)
        if obj is None:
            continue
        if inspect.isclass(obj) or inspect.isfunction(obj) or inspect.iscoroutinefunction(obj):
            symbols.append((name, obj))
    return symbols


def generate(module_name: str) -> str:
    symbols = collect_symbols(module_name)
    if not symbols:
        print(f"No public symbols found in {module_name!r}.", file=sys.stderr)
        sys.exit(1)

    parts: list[str] = []

    # Derive a test file path for the header comment
    test_path = "test/beta/test_" + module_name.rsplit(".", 1)[-1] + ".py"
    parts.append(_HEADER.format(test_path=test_path))
    parts.append(f"# Source module: {module_name}\n\n")

    for name, obj in symbols:
        slug = _snake(name)
        template = _ASYNC_STUB if _is_async(obj) else _SYNC_STUB
        parts.append(template.format(name=slug, symbol=name))

    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("module", help="Dotted module path, e.g. autogen.beta.exceptions")
    parser.add_argument("--out", metavar="FILE", help="Write output to this file instead of stdout")
    args = parser.parse_args()

    code = generate(args.module)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(code, encoding="utf-8")
        print(f"Wrote {out}")
    else:
        sys.stdout.write(code)


if __name__ == "__main__":
    main()
