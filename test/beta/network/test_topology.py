# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta.events import ModelMessage
from autogen.beta.network.primitives.envelope import Envelope
from autogen.beta.network.primitives.priority import DefaultPriority
from autogen.beta.network.topology import (
    BasePlugin,
    Conditional,
    Fanout,
    HubContext,
    Pipeline,
    RouteDecision,
)


class PassPlugin(BasePlugin):
    """Plugin that passes through and records calls."""

    def __init__(self, name: str = "pass"):
        self.name = name
        self.call_count = 0

    async def process(self, envelope, ctx):
        self.call_count += 1
        return envelope


class RejectPlugin(BasePlugin):
    """Plugin that rejects all envelopes."""

    async def process(self, envelope, ctx):
        return None


class ModifyPlugin(BasePlugin):
    """Plugin that modifies the envelope recipient."""

    def __init__(self, new_recipient: str):
        self._new_recipient = new_recipient

    async def process(self, envelope, ctx):
        envelope.recipient = self._new_recipient
        return envelope


def _make_envelope(**kwargs) -> Envelope:
    return Envelope(
        event=ModelMessage(content="test"),
        sender="source",
        **kwargs,
    )


class TestPipeline:
    @pytest.mark.asyncio
    async def test_sequential_processing(self) -> None:
        p1 = PassPlugin("p1")
        p2 = PassPlugin("p2")
        pipeline = Pipeline(p1, p2)

        ctx = HubContext(hub=None)  # type: ignore
        result = await pipeline.process(_make_envelope(), ctx)

        assert result is not None
        assert p1.call_count == 1
        assert p2.call_count == 1

    @pytest.mark.asyncio
    async def test_rejection_short_circuits(self) -> None:
        p1 = PassPlugin("p1")
        reject = RejectPlugin()
        p2 = PassPlugin("p2")
        pipeline = Pipeline(p1, reject, p2)

        ctx = HubContext(hub=None)  # type: ignore
        result = await pipeline.process(_make_envelope(), ctx)

        assert result is None
        assert p1.call_count == 1
        assert p2.call_count == 0  # Never reached

    @pytest.mark.asyncio
    async def test_modification_flows_through(self) -> None:
        modify = ModifyPlugin("new-target")
        check = PassPlugin("check")
        pipeline = Pipeline(modify, check)

        ctx = HubContext(hub=None)  # type: ignore
        result = await pipeline.process(_make_envelope(recipient="old-target"), ctx)

        assert result is not None
        assert result.recipient == "new-target"

    @pytest.mark.asyncio
    async def test_empty_pipeline(self) -> None:
        pipeline = Pipeline()
        ctx = HubContext(hub=None)  # type: ignore
        env = _make_envelope()
        result = await pipeline.process(env, ctx)
        assert result is env


class TestFanout:
    @pytest.mark.asyncio
    async def test_parallel_processing(self) -> None:
        p1 = PassPlugin("p1")
        p2 = PassPlugin("p2")
        fanout = Fanout(p1, p2)

        ctx = HubContext(hub=None)  # type: ignore
        result = await fanout.process(_make_envelope(), ctx)

        assert result is not None
        assert p1.call_count == 1
        assert p2.call_count == 1

    @pytest.mark.asyncio
    async def test_any_rejection_rejects(self) -> None:
        p1 = PassPlugin("p1")
        reject = RejectPlugin()
        fanout = Fanout(p1, reject)

        ctx = HubContext(hub=None)  # type: ignore
        result = await fanout.process(_make_envelope(), ctx)

        assert result is None

    @pytest.mark.asyncio
    async def test_rejection_preserves_additional_regardless_of_order(self) -> None:
        """Additional envelopes are preserved even when the rejecting plugin
        appears before the plugin that produces them."""

        class AdditionalPlugin(BasePlugin):
            async def process(self, envelope, ctx):
                return RouteDecision(
                    primary=envelope,
                    additional=[envelope.child(envelope.event, recipient="extra")],
                )

        # Reject is FIRST in the list — additional should still be preserved
        fanout = Fanout(RejectPlugin(), AdditionalPlugin())

        ctx = HubContext(hub=None)  # type: ignore
        result = await fanout.process(_make_envelope(), ctx)

        assert isinstance(result, RouteDecision)
        assert result.primary is None
        assert len(result.additional) == 1
        assert result.additional[0].recipient == "extra"


    @pytest.mark.asyncio
    async def test_mutation_safety(self) -> None:
        """Fanout gives each plugin a copy — mutations don't leak between plugins."""
        recipients_seen: list[str] = []

        class MutatePlugin(BasePlugin):
            def __init__(self, new_recipient: str):
                self._new_recipient = new_recipient

            async def process(self, envelope, ctx):
                recipients_seen.append(envelope.recipient)
                envelope.recipient = self._new_recipient
                return envelope

        p1 = MutatePlugin("from-p1")
        p2 = MutatePlugin("from-p2")
        fanout = Fanout(p1, p2)

        ctx = HubContext(hub=None)  # type: ignore
        original = _make_envelope(recipient="original")
        result = await fanout.process(original, ctx)

        # Both plugins should have seen the original recipient (not each other's mutation)
        assert all(r == "original" for r in recipients_seen)
        # Returned envelope is the original, unchanged
        assert result is original
        assert result.recipient == "original"


class TestConditional:
    @pytest.mark.asyncio
    async def test_true_branch(self) -> None:
        true_plugin = PassPlugin("true")
        false_plugin = PassPlugin("false")
        cond = Conditional(
            predicate=lambda env: env.priority == DefaultPriority.URGENT,
            if_true=Pipeline(true_plugin),
            if_false=Pipeline(false_plugin),
        )

        ctx = HubContext(hub=None)  # type: ignore
        result = await cond.process(_make_envelope(priority=DefaultPriority.URGENT), ctx)

        assert result is not None
        assert true_plugin.call_count == 1
        assert false_plugin.call_count == 0

    @pytest.mark.asyncio
    async def test_false_branch(self) -> None:
        true_plugin = PassPlugin("true")
        false_plugin = PassPlugin("false")
        cond = Conditional(
            predicate=lambda env: env.priority == DefaultPriority.URGENT,
            if_true=Pipeline(true_plugin),
            if_false=Pipeline(false_plugin),
        )

        ctx = HubContext(hub=None)  # type: ignore
        result = await cond.process(_make_envelope(priority=DefaultPriority.BACKGROUND), ctx)

        assert result is not None
        assert true_plugin.call_count == 0
        assert false_plugin.call_count == 1

    @pytest.mark.asyncio
    async def test_no_else_passes_through(self) -> None:
        true_plugin = PassPlugin("true")
        cond = Conditional(
            predicate=lambda env: env.priority == DefaultPriority.URGENT,
            if_true=Pipeline(true_plugin),
        )

        ctx = HubContext(hub=None)  # type: ignore
        env = _make_envelope(priority=DefaultPriority.BACKGROUND)
        result = await cond.process(env, ctx)

        assert result is env  # Pass-through
        assert true_plugin.call_count == 0


class TestComposition:
    @pytest.mark.asyncio
    async def test_nested_topologies(self) -> None:
        """Pipeline containing Fanout containing Pipeline — deep nesting works."""
        inner_pass = PassPlugin("inner")
        outer_pass = PassPlugin("outer")

        topology = Pipeline(
            outer_pass,
            Fanout(
                Pipeline(inner_pass),
                PassPlugin("side-effect"),
            ),
        )

        ctx = HubContext(hub=None)  # type: ignore
        result = await topology.process(_make_envelope(), ctx)

        assert result is not None
        assert outer_pass.call_count == 1
        assert inner_pass.call_count == 1


# =========================================================================
# RouteDecision tests
# =========================================================================


class MulticastPlugin(BasePlugin):
    """Plugin that returns RouteDecision with additional envelopes."""

    def __init__(self, targets: list[str]):
        self._targets = targets

    async def process(self, envelope, ctx):
        additional = [envelope.child(envelope.event, recipient=t) for t in self._targets]
        return RouteDecision(primary=envelope, additional=additional)


class RejectWithNotifyPlugin(BasePlugin):
    """Plugin that rejects primary but triggers additional delegations."""

    def __init__(self, notify_target: str):
        self._notify_target = notify_target

    async def process(self, envelope, ctx):
        return RouteDecision(
            primary=None,
            additional=[envelope.child(envelope.event, recipient=self._notify_target)],
        )


class TestRouteDecision:
    def test_basic_construction(self) -> None:
        env = _make_envelope(recipient="target")
        rd = RouteDecision(primary=env, additional=[])
        assert rd.primary is env
        assert rd.additional == []

    def test_reject_with_additional(self) -> None:
        env = _make_envelope(recipient="target")
        extra = env.child(env.event, recipient="alerter")
        rd = RouteDecision(primary=None, additional=[extra])
        assert rd.primary is None
        assert len(rd.additional) == 1
        assert rd.additional[0].recipient == "alerter"

    def test_defaults(self) -> None:
        rd = RouteDecision()
        assert rd.primary is None
        assert rd.additional == []


class TestPipelineWithRouteDecision:
    @pytest.mark.asyncio
    async def test_additional_envelopes_accumulate(self) -> None:
        """Additional envelopes from multiple plugins accumulate through pipeline."""
        p1 = MulticastPlugin(["agent_b"])
        p2 = MulticastPlugin(["agent_c"])
        pipeline = Pipeline(p1, p2)

        ctx = HubContext(hub=None)  # type: ignore
        result = await pipeline.process(_make_envelope(recipient="agent_a"), ctx)

        assert isinstance(result, RouteDecision)
        assert result.primary is not None
        assert result.primary.recipient == "agent_a"
        assert len(result.additional) == 2
        recipients = {e.recipient for e in result.additional}
        assert recipients == {"agent_b", "agent_c"}

    @pytest.mark.asyncio
    async def test_additional_bypass_remaining_plugins(self) -> None:
        """Additional envelopes from plugin 1 don't go through plugin 2."""
        call_log: list[str] = []

        class LoggingMulticast(BasePlugin):
            def __init__(self, name: str, targets: list[str]):
                self._name = name
                self._targets = targets

            async def process(self, envelope, ctx):
                call_log.append(f"{self._name}:{envelope.recipient}")
                additional = [envelope.child(envelope.event, recipient=t) for t in self._targets]
                return RouteDecision(primary=envelope, additional=additional)

        p1 = LoggingMulticast("p1", ["extra_1"])
        p2 = LoggingMulticast("p2", ["extra_2"])
        pipeline = Pipeline(p1, p2)

        ctx = HubContext(hub=None)  # type: ignore
        result = await pipeline.process(_make_envelope(recipient="primary"), ctx)

        # p1 sees "primary", p2 sees "primary" (not "extra_1")
        assert call_log == ["p1:primary", "p2:primary"]
        assert isinstance(result, RouteDecision)
        assert len(result.additional) == 2

    @pytest.mark.asyncio
    async def test_reject_preserves_collected_additional(self) -> None:
        """When plugin 2 rejects, additional from plugin 1 are still returned."""
        multicast = MulticastPlugin(["agent_b"])
        reject = RejectPlugin()
        pipeline = Pipeline(multicast, reject)

        ctx = HubContext(hub=None)  # type: ignore
        result = await pipeline.process(_make_envelope(recipient="agent_a"), ctx)

        assert isinstance(result, RouteDecision)
        assert result.primary is None
        assert len(result.additional) == 1
        assert result.additional[0].recipient == "agent_b"

    @pytest.mark.asyncio
    async def test_reject_with_notify_in_pipeline(self) -> None:
        """RejectWithNotifyPlugin rejects primary but returns additional."""
        plugin = RejectWithNotifyPlugin("alerter")
        after = PassPlugin("after")
        pipeline = Pipeline(plugin, after)

        ctx = HubContext(hub=None)  # type: ignore
        result = await pipeline.process(_make_envelope(recipient="target"), ctx)

        assert isinstance(result, RouteDecision)
        assert result.primary is None
        assert len(result.additional) == 1
        assert result.additional[0].recipient == "alerter"
        assert after.call_count == 0  # Never reached

    @pytest.mark.asyncio
    async def test_plain_envelope_return_still_works(self) -> None:
        """Plugins returning plain Envelope (no RouteDecision) work as before."""
        p1 = PassPlugin("p1")
        p2 = ModifyPlugin("new-target")
        pipeline = Pipeline(p1, p2)

        ctx = HubContext(hub=None)  # type: ignore
        result = await pipeline.process(_make_envelope(recipient="old"), ctx)

        assert not isinstance(result, RouteDecision)
        assert isinstance(result, Envelope)
        assert result.recipient == "new-target"

    @pytest.mark.asyncio
    async def test_mixed_plain_and_route_decision(self) -> None:
        """Pipeline with mix of plain-Envelope and RouteDecision plugins."""
        modify = ModifyPlugin("rerouted")
        multicast = MulticastPlugin(["extra"])
        check = PassPlugin("check")
        pipeline = Pipeline(modify, multicast, check)

        ctx = HubContext(hub=None)  # type: ignore
        result = await pipeline.process(_make_envelope(recipient="original"), ctx)

        assert isinstance(result, RouteDecision)
        assert result.primary.recipient == "rerouted"
        assert len(result.additional) == 1
        assert result.additional[0].recipient == "extra"
        assert check.call_count == 1


class TestFanoutWithRouteDecision:
    @pytest.mark.asyncio
    async def test_additional_collected_from_plugins(self) -> None:
        """Fanout collects additional envelopes from all plugins."""
        p1 = MulticastPlugin(["extra_1"])
        p2 = MulticastPlugin(["extra_2"])
        fanout = Fanout(p1, p2)

        ctx = HubContext(hub=None)  # type: ignore
        original = _make_envelope(recipient="primary")
        result = await fanout.process(original, ctx)

        assert isinstance(result, RouteDecision)
        assert result.primary is original  # Fanout returns original unchanged
        assert len(result.additional) == 2
        recipients = {e.recipient for e in result.additional}
        assert recipients == {"extra_1", "extra_2"}

    @pytest.mark.asyncio
    async def test_rejection_preserves_additional(self) -> None:
        """If any fanout plugin rejects, primary is None but accumulated
        additional envelopes are preserved (reject-with-side-effects)."""
        multicast = MulticastPlugin(["extra"])
        reject = RejectPlugin()
        fanout = Fanout(multicast, reject)

        ctx = HubContext(hub=None)  # type: ignore
        result = await fanout.process(_make_envelope(), ctx)
        # Primary rejected, but additional from multicast preserved
        assert isinstance(result, RouteDecision)
        assert result.primary is None
        assert len(result.additional) == 1
        assert result.additional[0].recipient == "extra"

    @pytest.mark.asyncio
    async def test_no_additional_returns_plain_envelope(self) -> None:
        """Fanout with no additional envelopes returns plain envelope."""
        p1 = PassPlugin("p1")
        p2 = PassPlugin("p2")
        fanout = Fanout(p1, p2)

        ctx = HubContext(hub=None)  # type: ignore
        original = _make_envelope()
        result = await fanout.process(original, ctx)

        assert result is original
        assert not isinstance(result, RouteDecision)


class TestConditionalWithRouteDecision:
    @pytest.mark.asyncio
    async def test_route_decision_passes_through(self) -> None:
        """Conditional transparently passes RouteDecision from branch."""
        multicast = MulticastPlugin(["extra"])
        cond = Conditional(
            predicate=lambda env: True,
            if_true=Pipeline(multicast),
        )

        ctx = HubContext(hub=None)  # type: ignore
        result = await cond.process(_make_envelope(recipient="primary"), ctx)

        assert isinstance(result, RouteDecision)
        assert result.primary.recipient == "primary"
        assert len(result.additional) == 1

    @pytest.mark.asyncio
    async def test_false_branch_route_decision(self) -> None:
        """False branch can also return RouteDecision."""
        multicast = MulticastPlugin(["extra"])
        cond = Conditional(
            predicate=lambda env: False,
            if_true=Pipeline(PassPlugin()),
            if_false=Pipeline(multicast),
        )

        ctx = HubContext(hub=None)  # type: ignore
        result = await cond.process(_make_envelope(recipient="primary"), ctx)

        assert isinstance(result, RouteDecision)
        assert len(result.additional) == 1


class TestCompositionWithRouteDecision:
    @pytest.mark.asyncio
    async def test_pipeline_fanout_pipeline_with_multicast(self) -> None:
        """Deep nesting: Pipeline(plugin, Fanout(multicast, pass), plugin)."""
        outer_before = PassPlugin("before")
        multicast = MulticastPlugin(["fanout_extra"])
        side_effect = PassPlugin("side")
        outer_after = PassPlugin("after")

        topology = Pipeline(
            outer_before,
            Fanout(multicast, side_effect),
            outer_after,
        )

        ctx = HubContext(hub=None)  # type: ignore
        result = await topology.process(_make_envelope(recipient="primary"), ctx)

        assert isinstance(result, RouteDecision)
        assert result.primary is not None
        assert len(result.additional) == 1
        assert result.additional[0].recipient == "fanout_extra"
        assert outer_before.call_count == 1
        assert outer_after.call_count == 1

    @pytest.mark.asyncio
    async def test_conditional_inside_pipeline_with_multicast(self) -> None:
        """Pipeline(pass, Conditional(multicast | pass), pass)."""
        multicast = MulticastPlugin(["cond_extra"])

        topology = Pipeline(
            PassPlugin("before"),
            Conditional(
                predicate=lambda env: env.recipient == "primary",
                if_true=Pipeline(multicast),
                if_false=Pipeline(PassPlugin("false")),
            ),
            PassPlugin("after"),
        )

        ctx = HubContext(hub=None)  # type: ignore
        result = await topology.process(_make_envelope(recipient="primary"), ctx)

        assert isinstance(result, RouteDecision)
        assert result.primary.recipient == "primary"
        assert len(result.additional) == 1
        assert result.additional[0].recipient == "cond_extra"

    @pytest.mark.asyncio
    async def test_multiple_multicast_plugins_accumulate(self) -> None:
        """Pipeline with 3 multicast plugins — all additional accumulate."""
        p1 = MulticastPlugin(["a"])
        p2 = MulticastPlugin(["b"])
        p3 = MulticastPlugin(["c"])
        pipeline = Pipeline(p1, p2, p3)

        ctx = HubContext(hub=None)  # type: ignore
        result = await pipeline.process(_make_envelope(recipient="primary"), ctx)

        assert isinstance(result, RouteDecision)
        assert result.primary.recipient == "primary"
        assert len(result.additional) == 3
        recipients = {e.recipient for e in result.additional}
        assert recipients == {"a", "b", "c"}

    @pytest.mark.asyncio
    async def test_trace_lineage_preserved_in_additional(self) -> None:
        """Additional envelopes inherit trace_id from the original."""
        multicast = MulticastPlugin(["extra"])
        pipeline = Pipeline(multicast)

        ctx = HubContext(hub=None)  # type: ignore
        original = _make_envelope(recipient="primary")
        result = await pipeline.process(original, ctx)

        assert isinstance(result, RouteDecision)
        extra = result.additional[0]
        assert extra.trace_id == original.trace_id
        assert extra.causation_id == original.correlation_id
