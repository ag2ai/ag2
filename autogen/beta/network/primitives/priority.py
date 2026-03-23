# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Priority primitives — mechanism for ordered delivery and conflict resolution.

The framework defines the protocol; developers define the policy.
Priority is consumed by Channel (for delivery ordering) and Hub (for routing
decisions). It is never on BaseEvent — it lives on the Envelope, at the
network level.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .envelope import Envelope


@runtime_checkable
class PriorityScheme(Protocol):
    """Defines how priorities are compared. Developer provides the policy."""

    def compare(self, a: Any, b: Any) -> int:
        """Negative if a < b, zero if equal, positive if a > b."""
        ...


@runtime_checkable
class ConflictResolver(Protocol):
    """Defines how conflicts between competing envelopes are resolved."""

    async def resolve(self, existing: Envelope, incoming: Envelope) -> Envelope:
        """Given two conflicting envelopes, return the winner."""
        ...


# ---------------------------------------------------------------------------
# Default implementations
# ---------------------------------------------------------------------------


class DefaultPriority(IntEnum):
    """Three-tier priority levels. Sensible default, override for custom needs."""

    BACKGROUND = 0
    NORMAL = 1
    URGENT = 2


class DefaultPriorityScheme:
    """Default PriorityScheme using the three-tier DefaultPriority levels.

    Example::

        scheme = DefaultPriorityScheme()
        scheme.compare(DefaultPriority.URGENT, DefaultPriority.BACKGROUND)  # > 0
    """

    BACKGROUND = DefaultPriority.BACKGROUND
    NORMAL = DefaultPriority.NORMAL
    URGENT = DefaultPriority.URGENT

    def compare(self, a: Any, b: Any) -> int:
        return int(a) - int(b)


class HighestPriorityWins:
    """Default conflict resolver: higher priority envelope wins."""

    async def resolve(self, existing: Envelope, incoming: Envelope) -> Envelope:
        if (incoming.priority is not None and existing.priority is not None) and (
            incoming.priority > existing.priority
        ):
            return incoming
        return existing
