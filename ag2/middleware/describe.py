# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from inspect import isroutine
from typing import Any, Protocol, runtime_checkable

__all__ = (
    "DescribableMiddleware",
    "DescribedMiddleware",
    "MiddlewareDescription",
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


def _describe(middleware: Any) -> MiddlewareDescription:
    """Describe ``middleware``, whether or not it implements ``describe()``.

    Internal. Middleware that opts in is asked directly; this exists so the
    library can describe anything uniformly, including middleware that did not
    opt in. It is not public API: callers reach descriptions through the
    surfaces that expose middleware.

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


@dataclass(frozen=True, slots=True)
class DescribedMiddleware:
    """A middleware paired with its description.

    Yielded wherever AG2 exposes attached middleware, so every entry reports
    itself whether or not that middleware opted in to :class:`DescribableMiddleware`.

    ``middleware`` is the object itself, not a copy, because a description
    cannot tell instances apart: one hook attached to ten tools and ten separate
    hooks configured identically produce equal descriptions.

    Identity answers "is this the same object", not "is this state shared".
    Registering a tool deep-copies it, which duplicates a class-based hook while
    leaving a plain function shared, so entries taken from two registered tools
    hold different objects even when one hook was attached to both.

    Entries are built on access, so ``agent.middleware[0] is agent.middleware[0]``
    is ``False``. Compare :attr:`middleware` rather than the entry.
    """

    middleware: Any

    @property
    def description(self) -> MiddlewareDescription:
        """What this middleware is and how it was configured."""

        return _describe(self.middleware)
