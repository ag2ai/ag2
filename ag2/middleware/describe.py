# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from inspect import isroutine
from typing import Any, Protocol, runtime_checkable

__all__ = (
    "DescribableMiddleware",
    "MiddlewareDescription",
    "describe_middleware",
)


@dataclass(frozen=True, slots=True)
class MiddlewareDescription:
    """A middleware's type and configuration.

    Supports equality but not hashing, since ``config`` may hold arbitrary
    values. Does not express which instances are shared between tools.

    Attributes:
        kind: Qualified name of the middleware type, without module path.
        config: Flat settings mapping. Excludes state that changes during a run.
        complete: ``False`` when the configuration was not fully captured.
            Forced to ``False`` if any entry in ``inner`` is incomplete.
        inner: Descriptions of wrapped middleware, for composites.

    See ADR 0013 for the rationale behind these choices.
    """

    kind: str
    config: dict[str, Any] = field(default_factory=dict)
    complete: bool = True
    inner: tuple["MiddlewareDescription", ...] = ()

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.complete and not all(described.complete for described in self.inner):
            object.__setattr__(self, "complete", False)


@runtime_checkable
class DescribableMiddleware(Protocol):
    """Middleware that can report its own type and configuration."""

    def describe(self) -> MiddlewareDescription: ...


def _kind_of(middleware: Any) -> str:
    """Qualified name of the middleware, without its module path."""

    if isinstance(middleware, type) or isroutine(middleware):
        return str(middleware.__qualname__)
    return type(middleware).__qualname__


def describe_middleware(middleware: Any) -> MiddlewareDescription:
    """Describe ``middleware``, whether or not it implements ``describe()``.

    Never raises. Middleware without ``describe()``, or whose ``describe()``
    raises or returns a non-``MiddlewareDescription``, yields ``complete=False``
    with an empty ``config``. Closure cells are not inspected.
    """

    describe = getattr(middleware, "describe", None)
    if callable(describe):
        try:
            described = describe()
        except Exception:
            described = None
        if isinstance(described, MiddlewareDescription):
            return described

    return MiddlewareDescription(kind=_kind_of(middleware), config={}, complete=False)
