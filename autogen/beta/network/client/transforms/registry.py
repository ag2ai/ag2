# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Per-:class:`ActorClient` registry of named transforms — Phase 5a.1.

Named transforms are the bridge between the terse ``apply: "redact_pii"``
rule syntax and long-lived stateful callables that live inside the
actor's address space. Every :class:`ActorClient` owns one
:class:`TransformRegistry` — the hub never holds any of these
callables, so tenant-registered named transforms never execute in the
hub process.

Entries are factories, not pre-instantiated instances. The pipeline
calls the factory once per ``(name, stage)`` slot, caches the result
for the lifetime of the current rule version, and drops the cached
instance on the next :class:`RuleChangedFrame` rebuild.
"""

from __future__ import annotations

from typing import Callable

from .protocol import Transform, TransformLookupError

__all__ = ("TransformFactory", "TransformRegistry")


TransformFactory = Callable[[], Transform]


class TransformRegistry:
    """Name → :class:`Transform` factory lookup.

    One per :class:`ActorClient`. Factories are zero-argument callables
    that return a fresh :class:`Transform` instance; the pipeline owns
    the lifecycle of those instances.
    """

    def __init__(self) -> None:
        self._factories: dict[str, TransformFactory] = {}

    def register(self, name: str, factory: TransformFactory) -> None:
        """Register a named factory. Replaces any prior entry."""

        self._factories[name] = factory

    def unregister(self, name: str) -> None:
        """Remove a named factory. No-op on missing names."""

        self._factories.pop(name, None)

    def has(self, name: str) -> bool:
        return name in self._factories

    def names(self) -> list[str]:
        return sorted(self._factories)

    def create(self, name: str) -> Transform:
        """Instantiate a fresh transform by name.

        Raises :class:`TransformLookupError` if the name is not
        registered — callers (the pipeline builder) catch this at
        rule-change time so a dangling reference surfaces immediately
        rather than when the first envelope tries to traverse the
        stage.
        """

        factory = self._factories.get(name)
        if factory is None:
            raise TransformLookupError(
                f"no named transform registered as {name!r}; "
                f"known: {self.names()}"
            )
        return factory()
