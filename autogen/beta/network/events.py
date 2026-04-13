# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""EventRegistry — hub-scoped catalog of stable event type names.

Phase 1 hard-coded a small set of ``ag2.*`` event type names
(``ag2.msg.text``, ``ag2.session.invite``, …) on the envelope wire
format. Phase 2 introduces an :class:`EventRegistry` so operators and
third-party adapters can register new stable names without forking the
envelope schema.

Design principles (§7):

* **Stable names.** Registered names are the wire format — not Python
  class paths. ``mycorp.audit.recorded`` is valid; a Python qualified
  name is not. Names survive refactors and cross-language boundaries.
* **Explicit registration.** No entry-point magic; operators call
  ``hub.register_event_type(name)`` or pass a pre-built registry to
  ``Hub(... event_registry=...)``. Matches the AuthAdapter + adapter
  registry extensibility stance (§3.2 / §5.5).
* **Built-ins pre-registered.** The default registry already contains
  every ``ag2.*`` name Phase 1 & Phase 2 use, so a fresh Hub without
  any registrations accepts them.
* **Unknown-type policy per hub.** The registry has a ``strict``
  flag: ``False`` (default) accepts unknown names (for forward
  compatibility with custom actors), ``True`` rejects them at
  envelope-post time.

The registry is consulted in two places in Phase 2:

1. ``Hub.register_event_type`` + ``register_event_types`` wrappers for
   operators.
2. ``Envelope`` post time: if ``event_registry.strict`` is True and
   the envelope's ``event_type`` is not registered, raise
   :class:`SessionTypeError`. Phase 3 WsLink tightens this to HTTP /
   WS wire validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .envelope import (
    EV_ERROR,
    EV_SESSION_CLOSED,
    EV_SESSION_INVITE,
    EV_SESSION_INVITE_ACK,
    EV_SESSION_INVITE_REJECT,
    EV_SESSION_OPENED,
    EV_TEXT,
    TASK_EVENT_TYPES,
)

# Built-in event type names — every event the Phase 1 & 2 & 4 surface emits.
BUILTIN_EVENT_TYPES: frozenset[str] = frozenset(
    {
        EV_TEXT,
        EV_SESSION_INVITE,
        EV_SESSION_INVITE_ACK,
        EV_SESSION_INVITE_REJECT,
        EV_SESSION_OPENED,
        EV_SESSION_CLOSED,
        EV_ERROR,
        # Phase 2 adapter events
        "ag2.auction.select",
    }
    | TASK_EVENT_TYPES  # Phase 4 network tasks
)


class UnknownEventTypeError(ValueError):
    """Raised when ``strict`` mode sees an unregistered event type."""


@dataclass
class EventTypeSpec:
    """Metadata the hub keeps about a registered event type.

    ``description`` is human-readable and surfaces in discovery APIs.
    ``allowed_in`` is an optional list of session-type names the event
    is valid in; empty means "any session". Phase 2 doesn't enforce
    ``allowed_in`` — it lives here so Phase 3 can wire it through
    ``SessionAdapter.validate_send`` without a schema change.
    """

    name: str
    description: str = ""
    allowed_in: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)


class EventRegistry:
    """Stable-name registry for envelope event types.

    Thread-safety: Phase 2 uses the registry from a single asyncio
    loop (the hub's connection handler), so no locking is needed.
    Phase 3 WsLink stays single-loop per hub process, so this remains
    correct there too.
    """

    def __init__(
        self,
        *,
        strict: bool = False,
        types: list[EventTypeSpec] | None = None,
    ) -> None:
        self._types: dict[str, EventTypeSpec] = {}
        self.strict = strict
        for name in BUILTIN_EVENT_TYPES:
            self._types[name] = EventTypeSpec(
                name=name, description="Built-in network event type"
            )
        for spec in types or []:
            self.register(spec)

    def register(self, spec: EventTypeSpec | str) -> EventTypeSpec:
        """Register a type. Accepts a string shorthand for convenience."""

        if isinstance(spec, str):
            spec = EventTypeSpec(name=spec)
        if not spec.name:
            raise ValueError("event type name must be a non-empty string")
        self._types[spec.name] = spec
        return spec

    def register_many(self, specs: list[EventTypeSpec | str]) -> list[EventTypeSpec]:
        return [self.register(s) for s in specs]

    def unregister(self, name: str) -> None:
        self._types.pop(name, None)

    def is_registered(self, name: str) -> bool:
        return name in self._types

    def get(self, name: str) -> EventTypeSpec | None:
        return self._types.get(name)

    def names(self) -> list[str]:
        return sorted(self._types)

    def check(self, name: str) -> None:
        """Validate an event type name against the current strictness.

        A permissive registry (``strict=False``) accepts anything —
        unknown names are forward-compatible custom types. A strict
        registry raises :class:`UnknownEventTypeError` on unknown
        names, so Phase 3 deployments can enforce a closed set of
        wire formats if they want to.
        """

        if self.strict and name not in self._types:
            raise UnknownEventTypeError(
                f"event type {name!r} is not registered on this hub " f"(strict mode)"
            )
