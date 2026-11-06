# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from ag2.exceptions import missing_optional_dependency

__all__ = ("LazyExport", "lazy_exports")


@dataclass(frozen=True, slots=True)
class LazyExport:
    """Where a lazily exported name lives.

    ``module`` is relative to the exporting package. With ``extra`` set, an
    ``ImportError`` becomes a ``missing_optional_dependency`` stand-in that hints
    ``pip install "ag2[<extra>]"``; without it the error propagates.
    """

    module: str
    extra: str | None = None


def lazy_exports(
    package: str,
    namespace: dict[str, Any],
    exports: Mapping[str, LazyExport],
) -> tuple[Callable[[str], Any], Callable[[], list[str]]]:
    """Build a package's PEP 562 ``__getattr__`` and ``__dir__`` for ``exports``.

    Each name is imported on first access and cached in ``namespace``, so later
    lookups never reach ``__getattr__``.
    """

    def getattr_(name: str) -> Any:
        try:
            export = exports[name]
        except KeyError:
            raise AttributeError(f"module {package!r} has no attribute {name!r}") from None

        try:
            value = getattr(import_module(export.module, package), name)
        except ImportError as e:
            if export.extra is None:
                raise
            value = missing_optional_dependency(name, export.extra, e)

        namespace[name] = value
        return value

    def dir_() -> list[str]:
        return sorted({*namespace, *exports})

    return getattr_, dir_
