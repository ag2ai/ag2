# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Topology — Plugin protocol, RouteDecision, and composable routing pipelines.

Plugins are the universal extension mechanism for the Hub. A Plugin has a
lifecycle (install/uninstall) and optionally processes envelopes in the
delegation path (process).

Two modes of operation:
1. System plugins — implement install/uninstall only. They subscribe to the
   Hub's stream, react to events, and manage resources independently.
2. Routing plugins — also implement process(). They sit in the delegation
   path and can transform, reject, or reroute envelopes.

Routing plugins return ``Envelope | RouteDecision | None``:
- ``Envelope`` — forward (possibly modified)
- ``RouteDecision`` — forward primary + trigger additional delegations
- ``None`` — reject

Topologies compose routing plugins into pipelines: Pipeline (sequential),
Fanout (parallel side-effects), Conditional (branching). Additional
envelopes from RouteDecision propagate upward through composition —
only the primary flows through the pipeline chain.
"""

from __future__ import annotations

import asyncio
import copy
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .primitives.envelope import Envelope

if TYPE_CHECKING:
    from .hub import Hub


# ---------------------------------------------------------------------------
# RouteDecision — structured routing outcome
# ---------------------------------------------------------------------------


@dataclass
class RouteDecision:
    """Structured routing outcome from a plugin.

    Separates the primary delegation from additional delegations triggered
    as side-effects. This is the primitive that enables multicast, co-routing,
    broadcast, and reject-with-notification patterns.

    The primary envelope (if not None) flows through the remaining topology
    pipeline and becomes the delegation that returns a result to the caller.

    Additional envelopes are dispatched by the Hub after the topology pipeline
    completes. Each goes through the full delegation path (depth tracking,
    topology, events) as an independent delegation.

    Examples::

        # Forward primary + replicate to two additional targets
        RouteDecision(
            primary=envelope,
            additional=[
                envelope.child(envelope.event, recipient="agent_b"),
                envelope.child(envelope.event, recipient="agent_c"),
            ],
        )

        # Reject primary but notify alerting agent
        RouteDecision(
            primary=None,
            additional=[envelope.child(alert_event, recipient="alerter")],
        )
    """

    primary: Envelope | None = None
    additional: list[Envelope] = field(default_factory=list)


#: Return type for Plugin.process() and Topology.process().
ProcessResult = Envelope | RouteDecision | None


# ---------------------------------------------------------------------------
# Helpers for normalizing ProcessResult
# ---------------------------------------------------------------------------


def _normalize(result: ProcessResult) -> tuple[Envelope | None, list[Envelope]]:
    """Split a ProcessResult into (primary, additional)."""
    if result is None:
        return None, []
    if isinstance(result, RouteDecision):
        return result.primary, list(result.additional)
    return result, []


def _to_result(primary: Envelope | None, additional: list[Envelope]) -> ProcessResult:
    """Reconstruct a ProcessResult from (primary, additional)."""
    if additional:
        return RouteDecision(primary=primary, additional=additional)
    return primary  # Envelope or None — backward-compatible form


# ---------------------------------------------------------------------------
# HubContext
# ---------------------------------------------------------------------------


class HubContext:
    """Context passed to plugins during envelope processing."""

    def __init__(self, hub: Hub) -> None:
        self.hub = hub


# ---------------------------------------------------------------------------
# Plugin protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Plugin(Protocol):
    """Extension point for the Hub.

    System plugins: implement install/uninstall only. Subscribe to hub.stream
    to monitor traffic, track metrics, or manage resources.

    Routing plugins: also implement process(). Sit in the delegation path
    and can transform, reject, or reroute envelopes.

    process() returns ``Envelope | RouteDecision | None``:
    - ``Envelope`` — forward (possibly modified)
    - ``RouteDecision`` — forward primary + trigger additional delegations
    - ``None`` — reject
    """

    def install(self, hub: Hub) -> None:
        """Called when plugin is added to Hub."""
        ...

    def uninstall(self) -> None:
        """Called when plugin is removed. Clean up subscriptions and resources."""
        ...

    async def process(self, envelope: Envelope, ctx: HubContext) -> ProcessResult:
        """Optional: process an envelope in the delegation path.

        Return Envelope to pass through (possibly modified), RouteDecision
        for multicast routing, or None to reject.
        Default: pass through unchanged.
        """
        return envelope


class BasePlugin:
    """Convenience base class for plugins. Pass-through process by default."""

    def install(self, hub: Hub) -> None:
        pass

    def uninstall(self) -> None:
        pass

    async def process(self, envelope: Envelope, ctx: HubContext) -> ProcessResult:
        return envelope


# ---------------------------------------------------------------------------
# Topologies — composable routing pipelines
# ---------------------------------------------------------------------------


class Topology(ABC):
    """Base for composable routing topologies.

    process() returns ``ProcessResult`` (Envelope | RouteDecision | None).
    Additional envelopes from RouteDecision propagate upward through
    composition — only the primary flows through the pipeline chain.
    """

    @abstractmethod
    async def process(self, envelope: Envelope, ctx: HubContext) -> ProcessResult:
        """Process an envelope through this topology."""
        ...

    def install_plugins(self, hub: Hub) -> None:
        """Install all plugins in this topology."""
        ...

    def uninstall_plugins(self) -> None:
        """Uninstall all plugins in this topology."""
        ...

    # Conform to Plugin protocol so nested topologies can be used as children
    # in Pipeline/Fanout without special-casing.
    def install(self, hub: Hub) -> None:
        self.install_plugins(hub)

    def uninstall(self) -> None:
        self.uninstall_plugins()


class Pipeline(Topology):
    """Sequential processing. Each plugin sees the output of the previous.

    Like nn.Sequential — transforms flow through in order.
    Any plugin returning None rejects the envelope (short-circuits).

    When a plugin returns a RouteDecision, the primary envelope continues
    through the remaining plugins. Additional envelopes accumulate and are
    included in the final result. If any plugin rejects the primary,
    accumulated additional envelopes are still returned (enabling
    reject-with-side-effects patterns).

    Example::

        topology = Pipeline(
            AuthPlugin(),
            RateLimiter(max_per_minute=10),
            TelemetryPlugin(),
        )
    """

    def __init__(self, *plugins: Plugin) -> None:
        self._plugins = list(plugins)

    async def process(self, envelope: Envelope, ctx: HubContext) -> ProcessResult:
        current: Envelope | None = envelope
        collected: list[Envelope] = []

        for plugin in self._plugins:
            if current is None:
                # Primary already rejected by a previous plugin.
                # Return collected additional envelopes if any (reject-with-side-effects).
                if collected:
                    return RouteDecision(primary=None, additional=collected)
                return None
            result = await plugin.process(current, ctx)
            primary, additional = _normalize(result)
            current = primary
            collected.extend(additional)

        return _to_result(current, collected)

    def install_plugins(self, hub: Hub) -> None:
        for plugin in self._plugins:
            plugin.install(hub)

    def uninstall_plugins(self) -> None:
        for plugin in self._plugins:
            plugin.uninstall()


class Fanout(Topology):
    """Parallel processing. All plugins see the same input concurrently.

    Useful for side-effects (logging, metrics) that don't modify the envelope.
    Returns the original envelope unchanged (side-effect only).
    If any plugin rejects the primary, the envelope is rejected.

    Additional envelopes from any plugin are collected and included in
    the result.

    Example::

        topology = Fanout(
            AuditLogger(),
            MetricsCollector(),
        )
    """

    def __init__(self, *plugins: Plugin) -> None:
        self._plugins = list(plugins)

    async def process(self, envelope: Envelope, ctx: HubContext) -> ProcessResult:
        results = await asyncio.gather(
            *(plugin.process(copy.deepcopy(envelope), ctx) for plugin in self._plugins),
            return_exceptions=True,
        )
        collected: list[Envelope] = []
        for r in results:
            if isinstance(r, BaseException):
                import logging
                logging.getLogger(__name__).exception(
                    "Fanout plugin raised during process", exc_info=r,
                )
                # Reject primary but preserve accumulated additional
                # (reject-with-side-effects, consistent with Pipeline).
                if collected:
                    return RouteDecision(primary=None, additional=collected)
                return None
            primary, additional = _normalize(r)
            if primary is None:
                # Reject primary but preserve accumulated additional.
                if collected:
                    return RouteDecision(primary=None, additional=collected)
                return None
            collected.extend(additional)

        if collected:
            return RouteDecision(primary=envelope, additional=collected)
        return envelope

    def install_plugins(self, hub: Hub) -> None:
        for plugin in self._plugins:
            plugin.install(hub)

    def uninstall_plugins(self) -> None:
        for plugin in self._plugins:
            plugin.uninstall()


class Conditional(Topology):
    """Branching. Route to different topologies based on a predicate.

    Passes through whatever the selected branch returns — Envelope,
    RouteDecision, or None. Composition is transparent.

    Example::

        topology = Conditional(
            predicate=lambda env: env.priority >= 2,
            if_true=Pipeline(AlertPlugin(), FastRouter()),
            if_false=Pipeline(QueuePlugin(), BatchRouter()),
        )
    """

    def __init__(
        self,
        predicate: Callable[[Envelope], bool],
        if_true: Topology,
        if_false: Topology | None = None,
    ) -> None:
        self._predicate = predicate
        self._if_true = if_true
        self._if_false = if_false

    async def process(self, envelope: Envelope, ctx: HubContext) -> ProcessResult:
        if self._predicate(envelope):
            return await self._if_true.process(envelope, ctx)
        elif self._if_false is not None:
            return await self._if_false.process(envelope, ctx)
        return envelope  # No else branch — pass through

    def install_plugins(self, hub: Hub) -> None:
        self._if_true.install_plugins(hub)
        if self._if_false is not None:
            self._if_false.install_plugins(hub)

    def uninstall_plugins(self) -> None:
        self._if_true.uninstall_plugins()
        if self._if_false is not None:
            self._if_false.uninstall_plugins()
