# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive priority tests — covers the full priority subsystem.

Tests cover:
- HighestPriorityWins with and without PriorityScheme
- PriorityChannel configurable default_priority
- Envelope.child() priority sentinel (_UNSET) behavior
- Hub auto-wiring PriorityChannel when priority_scheme is provided
- Hub._delegate() priority propagation to envelopes
- Hub.delegate() public API priority parameter
- Scheduler priority forwarding to Hub._delegate()
- Priority-based topology routing
- Edge cases: None priorities, mixed types, custom schemes
"""

import asyncio
from enum import IntEnum
from typing import Any
from uuid import uuid4

import pytest

from autogen.beta.events import BaseEvent, ModelMessage
from autogen.beta.network.hub import Hub
from autogen.beta.network.primitives.channel import LocalChannel, PriorityChannel
from autogen.beta.network.primitives.envelope import Envelope
from autogen.beta.network.primitives.priority import (
    ConflictResolver,
    DefaultPriority,
    DefaultPriorityScheme,
    HighestPriorityWins,
    PriorityScheme,
)
from autogen.beta.network.topology import BasePlugin, Pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _AskableAgent:
    """Mock agent that returns a canned result."""

    def __init__(self, name: str, result: str = "done"):
        self.name = name
        self._result = result
        self.received_messages: list[str] = []

    async def ask(self, message: str, **kwargs):
        self.received_messages.append(message)
        return type("Reply", (), {"content": self._result, "body": self._result})()


class _TrackingChannel:
    """Channel mock that records sent envelopes."""

    def __init__(self):
        self.sent: list[Envelope] = []
        self.closed = False

    async def send(self, envelope):
        self.sent.append(envelope)

    def subscribe(self, callback, *, condition=None):
        return uuid4()

    def unsubscribe(self, sub_id):
        pass

    async def close(self):
        self.closed = True


class _InvertedPriority(IntEnum):
    """Priority scheme where lower numeric value = higher priority (like P0/P1)."""

    P0_CRITICAL = 0
    P1_HIGH = 1
    P2_MEDIUM = 2
    P3_LOW = 3


class _InvertedScheme:
    """Custom scheme: lower numeric value = higher priority."""

    def compare(self, a: Any, b: Any) -> int:
        # Invert: P0 (0) is highest, so return negative of default
        return int(b) - int(a)


class _EnvelopeCapturingPlugin(BasePlugin):
    """Topology plugin that captures the envelope it processes."""

    def __init__(self):
        self.envelopes: list[Envelope] = []

    async def process(self, envelope: Envelope, ctx) -> Envelope | None:
        self.envelopes.append(envelope)
        return envelope


class _PriorityReroutePlugin(BasePlugin):
    """Topology plugin that reroutes based on envelope priority."""

    def __init__(self, threshold: int, high_target: str, low_target: str):
        self._threshold = threshold
        self._high_target = high_target
        self._low_target = low_target

    async def process(self, envelope: Envelope, ctx) -> Envelope | None:
        prio = envelope.priority
        if prio is not None and int(prio) >= self._threshold:
            envelope.recipient = self._high_target
        else:
            envelope.recipient = self._low_target
        return envelope


# =========================================================================
# HighestPriorityWins
# =========================================================================


class TestHighestPriorityWinsWithScheme:
    """HighestPriorityWins should use PriorityScheme.compare() when provided."""

    @pytest.mark.asyncio
    async def test_scheme_compare_used_for_resolution(self) -> None:
        """With a scheme, compare() determines the winner."""
        scheme = DefaultPriorityScheme()
        resolver = HighestPriorityWins(scheme=scheme)

        existing = Envelope(event=ModelMessage(content="low"), sender="a", priority=DefaultPriority.BACKGROUND)
        incoming = Envelope(event=ModelMessage(content="high"), sender="b", priority=DefaultPriority.URGENT)

        winner = await resolver.resolve(existing, incoming)
        assert winner is incoming

    @pytest.mark.asyncio
    async def test_scheme_lower_incoming_loses(self) -> None:
        scheme = DefaultPriorityScheme()
        resolver = HighestPriorityWins(scheme=scheme)

        existing = Envelope(event=ModelMessage(content="high"), sender="a", priority=DefaultPriority.URGENT)
        incoming = Envelope(event=ModelMessage(content="low"), sender="b", priority=DefaultPriority.BACKGROUND)

        winner = await resolver.resolve(existing, incoming)
        assert winner is existing

    @pytest.mark.asyncio
    async def test_scheme_equal_keeps_existing(self) -> None:
        scheme = DefaultPriorityScheme()
        resolver = HighestPriorityWins(scheme=scheme)

        existing = Envelope(event=ModelMessage(content="first"), sender="a", priority=DefaultPriority.NORMAL)
        incoming = Envelope(event=ModelMessage(content="second"), sender="b", priority=DefaultPriority.NORMAL)

        winner = await resolver.resolve(existing, incoming)
        assert winner is existing

    @pytest.mark.asyncio
    async def test_inverted_scheme_p0_beats_p3(self) -> None:
        """With an inverted scheme (P0 = highest), P0 incoming beats P3 existing."""
        resolver = HighestPriorityWins(scheme=_InvertedScheme())

        existing = Envelope(event=ModelMessage(content="low"), sender="a", priority=_InvertedPriority.P3_LOW)
        incoming = Envelope(event=ModelMessage(content="crit"), sender="b", priority=_InvertedPriority.P0_CRITICAL)

        winner = await resolver.resolve(existing, incoming)
        assert winner is incoming

    @pytest.mark.asyncio
    async def test_inverted_scheme_p3_loses_to_p0(self) -> None:
        """With inverted scheme, P3 incoming loses to P0 existing."""
        resolver = HighestPriorityWins(scheme=_InvertedScheme())

        existing = Envelope(event=ModelMessage(content="crit"), sender="a", priority=_InvertedPriority.P0_CRITICAL)
        incoming = Envelope(event=ModelMessage(content="low"), sender="b", priority=_InvertedPriority.P3_LOW)

        winner = await resolver.resolve(existing, incoming)
        assert winner is existing

    @pytest.mark.asyncio
    async def test_no_scheme_falls_back_to_gt(self) -> None:
        """Without a scheme, uses direct > comparison."""
        resolver = HighestPriorityWins()

        existing = Envelope(event=ModelMessage(content="low"), sender="a", priority=0)
        incoming = Envelope(event=ModelMessage(content="high"), sender="b", priority=10)

        winner = await resolver.resolve(existing, incoming)
        assert winner is incoming

    @pytest.mark.asyncio
    async def test_none_incoming_priority_keeps_existing(self) -> None:
        resolver = HighestPriorityWins(scheme=DefaultPriorityScheme())

        existing = Envelope(event=ModelMessage(content="has-prio"), sender="a", priority=DefaultPriority.NORMAL)
        incoming = Envelope(event=ModelMessage(content="no-prio"), sender="b", priority=None)

        winner = await resolver.resolve(existing, incoming)
        assert winner is existing

    @pytest.mark.asyncio
    async def test_none_existing_priority_keeps_existing(self) -> None:
        resolver = HighestPriorityWins(scheme=DefaultPriorityScheme())

        existing = Envelope(event=ModelMessage(content="no-prio"), sender="a", priority=None)
        incoming = Envelope(event=ModelMessage(content="has-prio"), sender="b", priority=DefaultPriority.URGENT)

        winner = await resolver.resolve(existing, incoming)
        assert winner is existing

    @pytest.mark.asyncio
    async def test_both_none_priority_keeps_existing(self) -> None:
        resolver = HighestPriorityWins()

        existing = Envelope(event=ModelMessage(content="first"), sender="a", priority=None)
        incoming = Envelope(event=ModelMessage(content="second"), sender="b", priority=None)

        winner = await resolver.resolve(existing, incoming)
        assert winner is existing


# =========================================================================
# PriorityChannel — default_priority
# =========================================================================


class TestPriorityChannelDefaultPriority:
    """PriorityChannel should use configurable default_priority for un-tagged envelopes."""

    @pytest.mark.asyncio
    async def test_default_priority_is_normal(self) -> None:
        """Without explicit default_priority, defaults to DefaultPriority.NORMAL."""
        channel = PriorityChannel()
        assert channel._default_priority == DefaultPriority.NORMAL

    @pytest.mark.asyncio
    async def test_custom_default_priority(self) -> None:
        """Explicit default_priority is stored and used."""
        channel = PriorityChannel(default_priority=42)
        assert channel._default_priority == 42

    @pytest.mark.asyncio
    async def test_none_priority_envelope_uses_default(self) -> None:
        """Envelope with priority=None uses the channel's default_priority for ordering."""
        received: list[str] = []

        async def handler(envelope, ctx):
            received.append(envelope.event.content)

        # default_priority=0 means un-tagged envelopes sort as BACKGROUND
        channel = PriorityChannel(default_priority=DefaultPriority.BACKGROUND)
        channel.subscribe(handler)

        # Send: un-tagged (will be BACKGROUND=0), then explicit URGENT
        await channel.send(Envelope(event=ModelMessage(content="untagged"), sender="a"))
        await channel.send(Envelope(event=ModelMessage(content="urgent"), sender="a", priority=DefaultPriority.URGENT))

        await asyncio.sleep(0.1)
        assert len(received) == 2
        # URGENT should come before BACKGROUND (untagged)
        urgent_idx = received.index("urgent")
        untagged_idx = received.index("untagged")
        assert urgent_idx < untagged_idx, f"Expected urgent before untagged, got: {received}"

    @pytest.mark.asyncio
    async def test_custom_default_for_custom_scheme(self) -> None:
        """Custom scheme with custom default_priority: un-tagged envelopes sort correctly."""

        class TradePriority(IntEnum):
            ANALYSIS = 0
            LIMIT_ORDER = 1
            MARKET_ORDER = 2
            RISK_ALERT = 3

        class TradePriorityScheme:
            def compare(self, a, b):
                return int(a) - int(b)

        received: list[str] = []

        async def handler(envelope, ctx):
            received.append(envelope.event.content)

        # Un-tagged envelopes should sort as ANALYSIS (lowest)
        channel = PriorityChannel(scheme=TradePriorityScheme(), default_priority=TradePriority.ANALYSIS)
        channel.subscribe(handler)

        await channel.send(Envelope(event=ModelMessage(content="untagged"), sender="a"))
        await channel.send(Envelope(event=ModelMessage(content="risk"), sender="a", priority=TradePriority.RISK_ALERT))

        await asyncio.sleep(0.1)
        assert len(received) == 2
        risk_idx = received.index("risk")
        untagged_idx = received.index("untagged")
        assert risk_idx < untagged_idx


class TestPriorityChannelWithInvertedScheme:
    """PriorityChannel should respect inverted schemes (lower number = higher priority)."""

    @pytest.mark.asyncio
    async def test_inverted_scheme_ordering(self) -> None:
        """With an inverted scheme, P0 (0) is delivered before P3 (3)."""
        received: list[str] = []

        async def handler(envelope, ctx):
            received.append(envelope.event.content)

        channel = PriorityChannel(
            scheme=_InvertedScheme(),
            default_priority=_InvertedPriority.P2_MEDIUM,
        )
        channel.subscribe(handler)

        await channel.send(Envelope(event=ModelMessage(content="p3-low"), sender="a", priority=_InvertedPriority.P3_LOW))
        await channel.send(Envelope(event=ModelMessage(content="p0-crit"), sender="a", priority=_InvertedPriority.P0_CRITICAL))

        await asyncio.sleep(0.1)
        assert len(received) == 2
        p0_idx = received.index("p0-crit")
        p3_idx = received.index("p3-low")
        assert p0_idx < p3_idx, f"Expected P0 before P3 with inverted scheme, got: {received}"


class TestPriorityChannelExpiredEnvelopes:
    """PriorityChannel should skip expired envelopes during drain."""

    @pytest.mark.asyncio
    async def test_expired_envelopes_are_skipped(self) -> None:
        received: list[str] = []

        async def handler(envelope, ctx):
            received.append(envelope.event.content)

        channel = PriorityChannel()
        channel.subscribe(handler)

        # Expired envelope (timestamp=0, ttl=0)
        await channel.send(
            Envelope(event=ModelMessage(content="expired"), sender="a", priority=DefaultPriority.URGENT, ttl=0.0, timestamp=0.0)
        )
        # Valid envelope
        await channel.send(
            Envelope(event=ModelMessage(content="valid"), sender="a", priority=DefaultPriority.NORMAL)
        )

        await asyncio.sleep(0.1)
        assert "valid" in received
        assert "expired" not in received

    @pytest.mark.asyncio
    async def test_closed_channel_rejects_sends(self) -> None:
        channel = PriorityChannel()
        await channel.close()
        with pytest.raises(RuntimeError, match="closed"):
            await channel.send(Envelope(event=ModelMessage(content="x"), sender="a"))


# =========================================================================
# Envelope.child() — priority sentinel
# =========================================================================


class TestEnvelopeChildPriority:
    """Envelope.child() should use _UNSET sentinel for priority like it does for recipient."""

    def test_child_inherits_priority_by_default(self) -> None:
        parent = Envelope(event=ModelMessage(content="p"), sender="a", priority=DefaultPriority.URGENT)
        child = parent.child(event=ModelMessage(content="c"))
        assert child.priority == DefaultPriority.URGENT

    def test_child_overrides_priority(self) -> None:
        parent = Envelope(event=ModelMessage(content="p"), sender="a", priority=DefaultPriority.URGENT)
        child = parent.child(event=ModelMessage(content="c"), priority=DefaultPriority.BACKGROUND)
        assert child.priority == DefaultPriority.BACKGROUND

    def test_child_clears_priority_to_none(self) -> None:
        """Passing priority=None explicitly should clear the child's priority."""
        parent = Envelope(event=ModelMessage(content="p"), sender="a", priority=DefaultPriority.URGENT)
        child = parent.child(event=ModelMessage(content="c"), priority=None)
        assert child.priority is None

    def test_child_inherits_none_priority(self) -> None:
        """Parent with no priority — child should also have None."""
        parent = Envelope(event=ModelMessage(content="p"), sender="a")
        child = parent.child(event=ModelMessage(content="c"))
        assert child.priority is None

    def test_child_sets_priority_when_parent_has_none(self) -> None:
        """Parent has no priority, child sets one."""
        parent = Envelope(event=ModelMessage(content="p"), sender="a")
        child = parent.child(event=ModelMessage(content="c"), priority=DefaultPriority.URGENT)
        assert child.priority == DefaultPriority.URGENT

    def test_child_inherits_trace_id(self) -> None:
        parent = Envelope(event=ModelMessage(content="p"), sender="a", priority=DefaultPriority.NORMAL)
        child = parent.child(event=ModelMessage(content="c"))
        assert child.trace_id == parent.trace_id

    def test_child_causation_id_points_to_parent(self) -> None:
        parent = Envelope(event=ModelMessage(content="p"), sender="a")
        child = parent.child(event=ModelMessage(content="c"))
        assert child.causation_id == parent.correlation_id

    def test_child_recipient_and_priority_independent(self) -> None:
        """Clearing priority and recipient independently should work."""
        parent = Envelope(
            event=ModelMessage(content="p"), sender="a", recipient="bob", priority=DefaultPriority.URGENT
        )
        child = parent.child(event=ModelMessage(content="c"), recipient=None, priority=None)
        assert child.recipient is None
        assert child.priority is None

    def test_child_integer_priority(self) -> None:
        """Plain integer priorities work too."""
        parent = Envelope(event=ModelMessage(content="p"), sender="a", priority=42)
        child = parent.child(event=ModelMessage(content="c"))
        assert child.priority == 42

        child_override = parent.child(event=ModelMessage(content="c"), priority=0)
        assert child_override.priority == 0


# =========================================================================
# Hub — auto-wiring PriorityChannel
# =========================================================================


class TestHubPriorityChannelWiring:
    """Hub should create PriorityChannel when priority_scheme is provided."""

    @pytest.mark.asyncio
    async def test_auto_creates_priority_channel(self) -> None:
        scheme = DefaultPriorityScheme()
        hub = Hub(priority_scheme=scheme)
        assert isinstance(hub._channel, PriorityChannel)

    @pytest.mark.asyncio
    async def test_no_scheme_uses_local_channel(self) -> None:
        hub = Hub()
        assert isinstance(hub._channel, LocalChannel)

    @pytest.mark.asyncio
    async def test_explicit_channel_overrides_auto_wiring(self) -> None:
        """If the user provides a channel, that takes precedence even with a scheme."""
        custom_channel = _TrackingChannel()
        hub = Hub(priority_scheme=DefaultPriorityScheme(), channel=custom_channel)
        assert hub._channel is custom_channel

    @pytest.mark.asyncio
    async def test_auto_wired_channel_uses_provided_scheme(self) -> None:
        """The auto-created PriorityChannel should use the Hub's priority_scheme."""
        scheme = DefaultPriorityScheme()
        hub = Hub(priority_scheme=scheme)
        assert isinstance(hub._channel, PriorityChannel)
        assert hub._channel._scheme is scheme

    @pytest.mark.asyncio
    async def test_priority_scheme_stored_as_property(self) -> None:
        scheme = DefaultPriorityScheme()
        hub = Hub(priority_scheme=scheme)
        assert hub.priority_scheme is scheme

    @pytest.mark.asyncio
    async def test_conflict_resolver_stored_as_property(self) -> None:
        resolver = HighestPriorityWins()
        hub = Hub(conflict_resolver=resolver)
        assert hub.conflict_resolver is resolver


# =========================================================================
# Hub._delegate() — priority on envelopes
# =========================================================================


class TestHubDelegatePriority:
    """Hub._delegate() should set priority on the delegation envelope."""

    @pytest.mark.asyncio
    async def test_delegate_sets_priority_on_envelope(self) -> None:
        channel = _TrackingChannel()
        hub = Hub(channel=channel)
        await hub.register(_AskableAgent("worker"))

        await hub._delegate("worker", "do work", source="caller", priority=DefaultPriority.URGENT)

        # First envelope is the DelegationRequest
        assert len(channel.sent) >= 1
        request_env = channel.sent[0]
        assert request_env.priority == DefaultPriority.URGENT

    @pytest.mark.asyncio
    async def test_delegate_no_priority_defaults_to_none(self) -> None:
        channel = _TrackingChannel()
        hub = Hub(channel=channel)
        await hub.register(_AskableAgent("worker"))

        await hub._delegate("worker", "do work", source="caller")

        request_env = channel.sent[0]
        assert request_env.priority is None

    @pytest.mark.asyncio
    async def test_delegate_priority_propagates_to_result_envelope(self) -> None:
        """Child envelope (result) should inherit priority from parent."""
        channel = _TrackingChannel()
        hub = Hub(channel=channel)
        await hub.register(_AskableAgent("worker", result="completed"))

        await hub._delegate("worker", "do work", source="caller", priority=DefaultPriority.URGENT)

        # Should have request and result envelopes
        assert len(channel.sent) == 2
        result_env = channel.sent[1]
        # Child inherits priority via Envelope.child()
        assert result_env.priority == DefaultPriority.URGENT

    @pytest.mark.asyncio
    async def test_delegate_custom_int_priority(self) -> None:
        channel = _TrackingChannel()
        hub = Hub(channel=channel)
        await hub.register(_AskableAgent("worker"))

        await hub._delegate("worker", "do work", source="caller", priority=99)

        assert channel.sent[0].priority == 99

    @pytest.mark.asyncio
    async def test_public_delegate_passes_priority(self) -> None:
        """Hub.delegate() (public API) should forward priority to _delegate()."""
        channel = _TrackingChannel()
        hub = Hub(channel=channel)
        await hub.register(_AskableAgent("worker"))

        await hub.delegate("caller", "worker", "do work", priority=DefaultPriority.URGENT)

        assert channel.sent[0].priority == DefaultPriority.URGENT

    @pytest.mark.asyncio
    async def test_public_delegate_no_priority(self) -> None:
        channel = _TrackingChannel()
        hub = Hub(channel=channel)
        await hub.register(_AskableAgent("worker"))

        await hub.delegate("caller", "worker", "do work")

        assert channel.sent[0].priority is None


# =========================================================================
# Hub — priority visible to topology
# =========================================================================


class TestHubTopologyPriority:
    """Topology plugins should see the priority on the envelope."""

    @pytest.mark.asyncio
    async def test_topology_sees_priority(self) -> None:
        capture = _EnvelopeCapturingPlugin()
        hub = Hub(topology=Pipeline(capture))
        await hub.register(_AskableAgent("worker"))

        await hub._delegate("worker", "task", source="src", priority=DefaultPriority.URGENT)

        assert len(capture.envelopes) == 1
        assert capture.envelopes[0].priority == DefaultPriority.URGENT

    @pytest.mark.asyncio
    async def test_topology_sees_none_priority(self) -> None:
        capture = _EnvelopeCapturingPlugin()
        hub = Hub(topology=Pipeline(capture))
        await hub.register(_AskableAgent("worker"))

        await hub._delegate("worker", "task", source="src")

        assert capture.envelopes[0].priority is None

    @pytest.mark.asyncio
    async def test_priority_based_routing(self) -> None:
        """Topology reroutes based on envelope priority."""
        router = _PriorityReroutePlugin(
            threshold=int(DefaultPriority.URGENT),
            high_target="senior",
            low_target="junior",
        )
        hub = Hub(topology=Pipeline(router))
        senior = _AskableAgent("senior", result="senior handled")
        junior = _AskableAgent("junior", result="junior handled")
        await hub.register(senior)
        await hub.register(junior)

        # URGENT -> senior
        result = await hub._delegate("junior", "critical task", source="triage", priority=DefaultPriority.URGENT)
        assert result == "senior handled"
        assert len(senior.received_messages) == 1

        # BACKGROUND -> junior
        result = await hub._delegate("senior", "cosmetic fix", source="triage", priority=DefaultPriority.BACKGROUND)
        assert result == "junior handled"
        assert len(junior.received_messages) == 1


# =========================================================================
# Scheduler — priority forwarding
# =========================================================================


class TestSchedulerPriorityForwarding:
    """Scheduler should pass entry.priority through to Hub._delegate()."""

    @pytest.mark.asyncio
    async def test_scheduled_watch_fires_with_priority(self) -> None:
        """When a watch fires, the scheduler should forward the entry's priority."""
        from unittest.mock import AsyncMock, patch

        from autogen.beta.scheduler import Scheduler
        from autogen.beta.watch import IntervalWatch

        hub = Hub()
        await hub.register(_AskableAgent("worker"))

        scheduler = Scheduler(hub=hub)
        watch_id = scheduler.add(
            IntervalWatch(9999),  # won't fire naturally
            target="worker",
            task="scheduled task",
            priority=DefaultPriority.URGENT,
        )

        # Verify priority is stored on the entry
        entry = scheduler._entries[watch_id]
        assert entry.priority == DefaultPriority.URGENT

        # Mock both stream.send (emits SchedulerTriggerFired) and _delegate
        # to isolate the priority forwarding test.
        from autogen.beta.context import Context

        mock_stream = AsyncMock()
        with (
            patch.object(hub.stream, "send", mock_stream),
            patch.object(hub, "_delegate", new_callable=AsyncMock, return_value="ok") as mock_delegate,
        ):
            scheduler._running = True
            ctx = Context(stream=hub.stream)
            await scheduler._handle_fire(entry, [], ctx)

            mock_delegate.assert_called_once_with(
                "worker", "scheduled task", source="scheduler", priority=DefaultPriority.URGENT
            )

    @pytest.mark.asyncio
    async def test_scheduled_watch_fires_without_priority(self) -> None:
        from unittest.mock import AsyncMock, patch

        from autogen.beta.context import Context
        from autogen.beta.scheduler import Scheduler
        from autogen.beta.watch import IntervalWatch

        hub = Hub()
        await hub.register(_AskableAgent("worker"))

        scheduler = Scheduler(hub=hub)
        watch_id = scheduler.add(
            IntervalWatch(9999),
            target="worker",
            task="task",
        )

        entry = scheduler._entries[watch_id]
        assert entry.priority is None

        mock_stream = AsyncMock()
        with (
            patch.object(hub.stream, "send", mock_stream),
            patch.object(hub, "_delegate", new_callable=AsyncMock, return_value="ok") as mock_delegate,
        ):
            scheduler._running = True
            ctx = Context(stream=hub.stream)
            await scheduler._handle_fire(entry, [], ctx)

            mock_delegate.assert_called_once_with(
                "worker", "task", source="scheduler", priority=None
            )

    @pytest.mark.asyncio
    async def test_scheduler_add_stores_priority(self) -> None:
        """Scheduler.add() should store the priority on the watch entry."""
        from autogen.beta.scheduler import Scheduler
        from autogen.beta.watch import IntervalWatch

        scheduler = Scheduler()

        w1 = scheduler.add(IntervalWatch(9999), target="x", task="t1", priority=DefaultPriority.URGENT)
        w2 = scheduler.add(IntervalWatch(9999), target="y", task="t2", priority=DefaultPriority.BACKGROUND)
        w3 = scheduler.add(IntervalWatch(9999), target="z", task="t3")  # no priority

        assert scheduler._entries[w1].priority == DefaultPriority.URGENT
        assert scheduler._entries[w2].priority == DefaultPriority.BACKGROUND
        assert scheduler._entries[w3].priority is None


# =========================================================================
# PriorityScheme protocol conformance
# =========================================================================


class TestPrioritySchemeProtocol:
    """Verify that custom classes satisfy the PriorityScheme protocol."""

    def test_default_scheme_is_protocol_compliant(self) -> None:
        assert isinstance(DefaultPriorityScheme(), PriorityScheme)

    def test_inverted_scheme_is_protocol_compliant(self) -> None:
        assert isinstance(_InvertedScheme(), PriorityScheme)

    def test_lambda_scheme_is_not_protocol_compliant(self) -> None:
        """A plain lambda doesn't satisfy the protocol (no compare method)."""
        assert not isinstance(lambda a, b: a - b, PriorityScheme)


class TestConflictResolverProtocol:
    """Verify ConflictResolver protocol conformance."""

    def test_highest_priority_wins_is_protocol_compliant(self) -> None:
        assert isinstance(HighestPriorityWins(), ConflictResolver)


# =========================================================================
# DefaultPriorityScheme edge cases
# =========================================================================


class TestDefaultPrioritySchemeEdgeCases:
    def test_compare_raw_ints(self) -> None:
        scheme = DefaultPriorityScheme()
        assert scheme.compare(100, 0) == 100
        assert scheme.compare(0, 100) == -100
        assert scheme.compare(50, 50) == 0

    def test_compare_negative_values(self) -> None:
        scheme = DefaultPriorityScheme()
        assert scheme.compare(-1, 0) < 0
        assert scheme.compare(0, -1) > 0

    def test_scheme_class_attributes(self) -> None:
        scheme = DefaultPriorityScheme()
        assert scheme.BACKGROUND == DefaultPriority.BACKGROUND
        assert scheme.NORMAL == DefaultPriority.NORMAL
        assert scheme.URGENT == DefaultPriority.URGENT


# =========================================================================
# End-to-end: Hub + PriorityChannel + topology
# =========================================================================


class TestPriorityEndToEnd:
    """Integration tests combining Hub, PriorityChannel, and topology."""

    @pytest.mark.asyncio
    async def test_hub_with_priority_scheme_and_topology_routing(self) -> None:
        """Full stack: priority on envelope, PriorityChannel delivery, topology rerouting."""
        router = _PriorityReroutePlugin(
            threshold=int(DefaultPriority.URGENT),
            high_target="senior",
            low_target="junior",
        )
        hub = Hub(
            priority_scheme=DefaultPriorityScheme(),
            conflict_resolver=HighestPriorityWins(scheme=DefaultPriorityScheme()),
            topology=Pipeline(router),
        )
        # Verify auto-wired PriorityChannel
        assert isinstance(hub._channel, PriorityChannel)

        senior = _AskableAgent("senior", result="senior done")
        junior = _AskableAgent("junior", result="junior done")
        await hub.register(senior)
        await hub.register(junior)

        # URGENT task -> rerouted to senior
        result = await hub.delegate("triage", "junior", "CRITICAL: outage", priority=DefaultPriority.URGENT)
        assert result == "senior done"

        # BACKGROUND task -> rerouted to junior
        result = await hub.delegate("triage", "senior", "LOW: typo fix", priority=DefaultPriority.BACKGROUND)
        assert result == "junior done"

        await hub.close()

    @pytest.mark.asyncio
    async def test_inverted_scheme_end_to_end(self) -> None:
        """Custom inverted scheme works through the full stack."""
        capture = _EnvelopeCapturingPlugin()
        hub = Hub(
            priority_scheme=_InvertedScheme(),
            topology=Pipeline(capture),
        )
        assert isinstance(hub._channel, PriorityChannel)

        await hub.register(_AskableAgent("worker"))

        await hub.delegate("src", "worker", "P0 task", priority=_InvertedPriority.P0_CRITICAL)
        assert capture.envelopes[0].priority == _InvertedPriority.P0_CRITICAL

        await hub.close()

    @pytest.mark.asyncio
    async def test_multiple_delegations_all_carry_priority(self) -> None:
        """Multiple sequential delegations each carry their own priority."""
        channel = _TrackingChannel()
        hub = Hub(channel=channel)
        await hub.register(_AskableAgent("w1"))
        await hub.register(_AskableAgent("w2"))

        await hub.delegate("src", "w1", "task1", priority=DefaultPriority.URGENT)
        await hub.delegate("src", "w2", "task2", priority=DefaultPriority.BACKGROUND)
        await hub.delegate("src", "w1", "task3")  # no priority

        # Request envelopes are at indices 0, 2, 4 (result envelopes at 1, 3, 5)
        assert channel.sent[0].priority == DefaultPriority.URGENT
        assert channel.sent[2].priority == DefaultPriority.BACKGROUND
        assert channel.sent[4].priority is None

        await hub.close()


# =========================================================================
# Envelope wire format — priority round-trip
# =========================================================================


class TestEnvelopePriorityWireFormat:
    """Priority should survive serialization round-trips."""

    def test_to_dict_includes_priority(self) -> None:
        env = Envelope(event=ModelMessage(content="x"), sender="a", priority=DefaultPriority.URGENT)
        d = env.to_dict()
        assert d["priority"] == DefaultPriority.URGENT

    def test_to_dict_none_priority(self) -> None:
        env = Envelope(event=ModelMessage(content="x"), sender="a")
        d = env.to_dict()
        assert d["priority"] is None

    def test_from_dict_restores_priority(self) -> None:
        env = Envelope(event=ModelMessage(content="x"), sender="a", priority=2)
        d = env.to_dict()
        restored = Envelope.from_dict(d)
        assert restored.priority == 2

    def test_from_dict_none_priority(self) -> None:
        env = Envelope(event=ModelMessage(content="x"), sender="a")
        d = env.to_dict()
        restored = Envelope.from_dict(d)
        assert restored.priority is None

    def test_round_trip_custom_priority(self) -> None:
        """Non-standard priority values survive round-trip."""
        env = Envelope(event=ModelMessage(content="x"), sender="a", priority=999)
        restored = Envelope.from_dict(env.to_dict())
        assert restored.priority == 999
