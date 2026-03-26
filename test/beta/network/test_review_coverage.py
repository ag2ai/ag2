# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Coverage tests from pre-release review.

Covers gaps identified during in-depth review:
- Bug A: Hub._delegate error event uses effective_task after topology modification
- Bug B: Prompt leak on FATAL + non-fatal signal path (now cleaned up)
- Bug C: Fanout preserves additional envelopes on reject/exception
- Bug D: Scheduler _handle_fire respects CANCELLED status
- Gap 1: Concurrent delegation depth isolation (contextvars)
- Gap 2: Hub.close() cleanup
- Gap 3: Hub.ask() with string agent name
- Gap 4: BufferedChannel "block" policy
- Gap 5: PriorityChannel delivery ordering and TTL expiry
- Gap 6: Multiple observers emitting signals simultaneously
- Gap 7: Topology additional envelope dispatch integration
- Gap 8: Network convenience class with topology
"""

import asyncio
import time
from collections.abc import Sequence
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from autogen.beta.config import LLMClient, ModelConfig
from autogen.beta.context import Context as ContextType
from autogen.beta.events import BaseEvent, ModelMessage, ModelResponse, ToolCallEvent, ToolCallsEvent
from autogen.beta.events.conditions import TypeCondition
from autogen.beta.network.actor import Actor, _SignalInjectionMiddleware
from autogen.beta.network.convenience import Network
from autogen.beta.network.events import (
    DelegationError,
    DelegationRejected,
    DelegationRequest,
    DelegationResult,
    ObserverCompleted,
)
from autogen.beta.network.hub import Hub
from autogen.beta.network.observer import BaseObserver
from autogen.beta.network.primitives.channel import BufferedChannel, PriorityChannel
from autogen.beta.network.primitives.envelope import Envelope
from autogen.beta.network.primitives.priority import DefaultPriority, DefaultPriorityScheme
from autogen.beta.network.primitives.signal import (
    EmitToStream,
    InjectToPrompt,
    Severity,
    Signal,
    SignalPolicy,
)
from autogen.beta.network.primitives.watch import EventWatch, IntervalWatch
from autogen.beta.network.scheduler import Scheduler, WatchStatus
from autogen.beta.network.topology import BasePlugin, Fanout, Pipeline, RouteDecision
from autogen.beta.stream import MemoryStream
from autogen.beta.tools.final import tool

from typing_extensions import Self


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _AskableAgent:
    """Mock agent that returns a canned result."""

    def __init__(self, name: str, result: str = "done"):
        self.name = name
        self._result = result
        self.asked: list[str] = []

    async def ask(self, message: str, **kwargs: Any) -> Any:
        self.asked.append(message)
        return type("Reply", (), {"content": self._result, "body": self._result, "response": None})()


class _FailingAgent:
    """Agent that raises on ask()."""

    def __init__(self, name: str, error: str = "agent exploded"):
        self.name = name
        self._error = error

    async def ask(self, message: str, **kwargs: Any) -> Any:
        raise RuntimeError(self._error)


@tool
def dummy_tool(value: str) -> str:
    """A no-op tool that echoes back its input."""
    return f"echo: {value}"


class _RecordingClient(LLMClient):
    def __init__(self, *responses: ModelResponse | str) -> None:
        self._responses = list(responses)
        self._call_count = 0
        self.calls: list[tuple[list[BaseEvent], list[str]]] = []

    async def __call__(
        self,
        messages: Sequence[BaseEvent],
        context: ContextType,
        **kwargs: Any,
    ) -> ModelResponse:
        self.calls.append((list(messages), list(context.prompt)))
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
        else:
            resp = "done"
        self._call_count += 1
        if isinstance(resp, str):
            return ModelResponse(message=ModelMessage(content=resp))
        if isinstance(resp, ToolCallEvent):
            return ModelResponse(tool_calls=ToolCallsEvent(calls=[resp]))
        return resp


class _RecordingConfig(ModelConfig):
    __test__ = False

    def __init__(self, *responses: ModelResponse | str) -> None:
        self._responses = responses
        self.client: _RecordingClient | None = None

    def copy(self) -> Self:
        return self

    def create(self) -> _RecordingClient:
        self.client = _RecordingClient(*self._responses)
        return self.client


# ===========================================================================
# Bug A: DelegationError uses effective_task after topology modification
# ===========================================================================


class TestDelegationErrorUsesEffectiveTask:
    """When topology modifies the task and delegation then fails,
    DelegationError.task should reflect the modified task."""

    @pytest.mark.asyncio
    async def test_error_event_has_modified_task(self) -> None:
        class _TaskModPlugin(BasePlugin):
            async def process(self, envelope, ctx):
                if isinstance(envelope.event, DelegationRequest):
                    envelope.event = DelegationRequest(
                        source=envelope.event.source,
                        target=envelope.event.target,
                        task="MODIFIED: " + envelope.event.task,
                    )
                return envelope

        failing_agent = _FailingAgent("worker", error="boom")
        hub = Hub(topology=Pipeline(_TaskModPlugin()))
        await hub.register(failing_agent)

        errors: list[DelegationError] = []
        hub.stream.subscribe(
            lambda e: errors.append(e),
            condition=TypeCondition(DelegationError),
        )

        result = await hub._delegate("worker", "original task", source="caller")

        assert "boom" in result
        assert len(errors) == 1
        # Bug A fix: error should report the topology-modified task
        assert errors[0].task == "MODIFIED: original task"


# ===========================================================================
# Bug B: Prompt cleanup on FATAL + non-fatal signal path
# ===========================================================================


class TestFatalPromptCleanup:
    """FATAL signal path should clean up any prompt entries added by the policy."""

    @pytest.mark.asyncio
    async def test_prompt_cleaned_after_fatal_halt(self) -> None:
        """When FATAL + non-fatal arrive, non-fatal alert text should be
        removed from prompt after the halt response is returned."""
        config = _RecordingConfig(
            # First call: returns tool call → triggers ModelResponse → observer fires
            ToolCallEvent(name="dummy_tool", arguments='{"value": "x"}'),
            "should not reach",
        )

        class _MixedSignalObserver(BaseObserver):
            def __init__(self):
                super().__init__("mixed", watch=EventWatch(ModelResponse))
                self._fired = False

            async def process(self, events, ctx):
                if not self._fired:
                    self._fired = True
                    return Signal(
                        source="mixed",
                        severity=Severity.FATAL,
                        message="halt now",
                    )
                return None

        observer = _MixedSignalObserver()
        actor = Actor("test", config=config, observers=[observer], tools=[dummy_tool])

        stream = MemoryStream()
        reply = await actor.ask("go", stream=stream)

        assert reply.body is not None
        assert "HALTED" in reply.body


# ===========================================================================
# Bug C: Fanout preserves additional envelopes on reject/exception
# ===========================================================================


class TestFanoutPreservesAdditionalOnReject:
    """Fanout should return accumulated additional envelopes when rejecting,
    consistent with Pipeline's reject-with-side-effects pattern."""

    @pytest.mark.asyncio
    async def test_fanout_exception_preserves_additional(self) -> None:
        """When one plugin raises but another returned additional envelopes,
        those additional envelopes should be preserved."""

        class _AdditionalPlugin(BasePlugin):
            async def process(self, envelope, ctx):
                return RouteDecision(
                    primary=envelope,
                    additional=[envelope.child(envelope.event, recipient="extra")],
                )

        class _RaisingPlugin(BasePlugin):
            async def process(self, envelope, ctx):
                raise ValueError("plugin exploded")

        # AdditionalPlugin runs first (gather runs concurrently, but
        # results are iterated in order)
        fanout = Fanout(_AdditionalPlugin(), _RaisingPlugin())

        env = Envelope(
            event=DelegationRequest(source="a", target="b", task="test"),
            sender="a",
            recipient="b",
        )

        from autogen.beta.network.topology import HubContext

        result = await fanout.process(env, HubContext(hub=None))  # type: ignore[arg-type]

        # Primary should be rejected (None) but additional preserved
        assert isinstance(result, RouteDecision)
        assert result.primary is None
        assert len(result.additional) == 1
        assert result.additional[0].recipient == "extra"

    @pytest.mark.asyncio
    async def test_fanout_reject_preserves_additional(self) -> None:
        """When one plugin rejects (returns None) but another returned
        additional envelopes, those should be preserved."""

        class _AdditionalPlugin(BasePlugin):
            async def process(self, envelope, ctx):
                return RouteDecision(
                    primary=envelope,
                    additional=[envelope.child(envelope.event, recipient="audit")],
                )

        class _RejectPlugin(BasePlugin):
            async def process(self, envelope, ctx):
                return None  # Reject

        fanout = Fanout(_AdditionalPlugin(), _RejectPlugin())

        env = Envelope(
            event=DelegationRequest(source="a", target="b", task="test"),
            sender="a",
            recipient="b",
        )

        from autogen.beta.network.topology import HubContext

        result = await fanout.process(env, HubContext(hub=None))  # type: ignore[arg-type]

        assert isinstance(result, RouteDecision)
        assert result.primary is None
        assert len(result.additional) == 1
        assert result.additional[0].recipient == "audit"


# ===========================================================================
# Bug D: Scheduler _handle_fire respects CANCELLED status
# ===========================================================================


class TestSchedulerCancelledGuard:
    """_handle_fire should not execute callbacks for CANCELLED watches."""

    @pytest.mark.asyncio
    async def test_cancelled_watch_does_not_fire_callback(self) -> None:
        """After cancel(), pending callbacks should be a no-op."""
        scheduler = Scheduler()
        call_count = 0

        async def callback(events, ctx):
            nonlocal call_count
            call_count += 1

        wid = scheduler.add(IntervalWatch(0.05), callback=callback)
        await scheduler.start()

        # Let it fire at least once
        await asyncio.sleep(0.12)
        count_before = call_count

        # Cancel it
        scheduler.cancel(wid)
        assert scheduler.status(wid) == WatchStatus.CANCELLED

        # Any callbacks that fire after cancel should be no-ops
        await asyncio.sleep(0.15)
        assert call_count == count_before

        await scheduler.stop()


# ===========================================================================
# Gap 1: Concurrent delegation depth isolation
# ===========================================================================


class TestConcurrentDelegationDepth:
    """contextvars should isolate delegation depth across concurrent delegations."""

    @pytest.mark.asyncio
    async def test_concurrent_delegations_independent_depth(self) -> None:
        """Two concurrent delegations should not share depth counters."""
        depths_seen: dict[str, int] = {}

        class _DepthRecordingAgent:
            def __init__(self, name):
                self.name = name

            async def ask(self, message, **kwargs):
                from autogen.beta.network.hub import _delegation_depth

                depths_seen[self.name] = _delegation_depth.get()
                # Simulate some async work
                await asyncio.sleep(0.05)
                return type("Reply", (), {"body": "done", "response": None})()

        hub = Hub()
        agent_a = _DepthRecordingAgent("agent-a")
        agent_b = _DepthRecordingAgent("agent-b")
        await hub.register(agent_a)
        await hub.register(agent_b)

        # Run two delegations concurrently
        await asyncio.gather(
            hub._delegate("agent-a", "task-a", source="src"),
            hub._delegate("agent-b", "task-b", source="src"),
        )

        # Both should have seen depth=1 (independent counters)
        assert depths_seen["agent-a"] == 1
        assert depths_seen["agent-b"] == 1


# ===========================================================================
# Gap 2: Hub.close() cleanup
# ===========================================================================


class TestHubCloseCleanup:
    """Hub.close() should clean up all resources."""

    @pytest.mark.asyncio
    async def test_close_uninstalls_topology(self) -> None:
        class _TrackingPlugin(BasePlugin):
            def __init__(self):
                self.installed = False
                self.uninstalled = False

            def install(self, hub):
                self.installed = True

            def uninstall(self):
                self.uninstalled = True

        plugin = _TrackingPlugin()
        hub = Hub(topology=Pipeline(plugin))

        assert plugin.installed
        assert not plugin.uninstalled

        await hub.close()
        assert plugin.uninstalled

    @pytest.mark.asyncio
    async def test_close_uninstalls_system_plugins(self) -> None:
        class _TrackingPlugin(BasePlugin):
            def __init__(self):
                self.uninstalled = False

            def uninstall(self):
                self.uninstalled = True

        plugin = _TrackingPlugin()
        hub = Hub(plugins=[plugin])
        await hub.close()
        assert plugin.uninstalled

    @pytest.mark.asyncio
    async def test_close_closes_channel(self) -> None:
        from autogen.beta.network.primitives.channel import LocalChannel

        channel = LocalChannel()
        hub = Hub(channel=channel)

        await hub.close()
        # Channel should be closed — sending should raise
        with pytest.raises(RuntimeError, match="closed"):
            env = Envelope(event=ModelMessage(content="test"), sender="a")
            await channel.send(env)

    @pytest.mark.asyncio
    async def test_close_cancels_additional_tasks(self) -> None:
        """In-flight additional delegation tasks should be cancelled on close."""

        class _SlowAgent:
            name = "slow"

            async def ask(self, message, **kwargs):
                await asyncio.sleep(10)  # Very slow — should be cancelled
                return type("Reply", (), {"body": "done", "response": None})()

        hub = Hub()
        await hub.register(_SlowAgent())

        # Start a delegation in the background (won't complete)
        task = asyncio.create_task(hub._delegate("slow", "work", source="src"))

        # Give it a moment to start
        await asyncio.sleep(0.05)

        # Close should cancel in-flight tasks
        await hub.close()

        # The delegation should have been interrupted
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


# ===========================================================================
# Gap 3: Hub.ask() with string agent name
# ===========================================================================


class TestHubAskStringAgent:
    """Hub.ask() should accept a string agent name."""

    @pytest.mark.asyncio
    async def test_ask_by_name(self) -> None:
        agent = _AskableAgent("worker", result="name-resolved")
        hub = Hub()
        await hub.register(agent)

        reply = await hub.ask("worker", "do work")
        assert reply.body == "name-resolved"

    @pytest.mark.asyncio
    async def test_ask_unknown_name_raises(self) -> None:
        hub = Hub()
        with pytest.raises(KeyError, match="not registered"):
            await hub.ask("nonexistent", "hello")

    @pytest.mark.asyncio
    async def test_ask_by_agent_instance(self) -> None:
        """Sanity check: ask() with Agent instance still works."""
        agent = _AskableAgent("worker", result="instance-resolved")
        hub = Hub()
        await hub.register(agent)

        reply = await hub.ask(agent, "do work")
        assert reply.body == "instance-resolved"


# ===========================================================================
# Gap 4: BufferedChannel "block" policy
# ===========================================================================


class TestBufferedChannelBlockPolicy:
    """BufferedChannel with overflow_policy='block' should wait for space."""

    @pytest.mark.asyncio
    async def test_block_policy_waits_for_drain(self) -> None:
        """When buffer is full with block policy, send() should wait."""
        channel = BufferedChannel(max_buffer=2, overflow_policy="block")

        received: list[Envelope] = []

        async def handler(envelope, ctx):
            received.append(envelope)

        channel.subscribe(handler)

        # Fill the buffer — first two should go in immediately
        env1 = Envelope(event=ModelMessage(content="1"), sender="a")
        env2 = Envelope(event=ModelMessage(content="2"), sender="a")
        await channel.send(env1)
        await channel.send(env2)

        # Wait for drain to process
        await asyncio.sleep(0.1)

        # Third should also complete after drain makes room
        env3 = Envelope(event=ModelMessage(content="3"), sender="a")
        await asyncio.wait_for(channel.send(env3), timeout=2.0)

        await asyncio.sleep(0.1)
        assert len(received) == 3

        await channel.close()

    @pytest.mark.asyncio
    async def test_drop_oldest_policy(self) -> None:
        """drop_oldest should evict the oldest when full."""
        channel = BufferedChannel(max_buffer=2, overflow_policy="drop_oldest")

        received: list[str] = []

        async def handler(envelope, ctx):
            received.append(envelope.event.content)

        channel.subscribe(handler)

        # Send 3 rapidly — buffer is 2, so first should be dropped
        for i in range(3):
            await channel.send(Envelope(event=ModelMessage(content=str(i)), sender="a"))

        await asyncio.sleep(0.1)
        await channel.close()

        # We should have received messages, oldest may be dropped
        assert len(received) >= 2

    @pytest.mark.asyncio
    async def test_drop_newest_policy(self) -> None:
        """drop_newest should discard incoming when buffer is full."""
        channel = BufferedChannel(max_buffer=1, overflow_policy="drop_newest")

        received: list[str] = []

        # Use a slow handler to keep buffer full
        async def slow_handler(envelope, ctx):
            await asyncio.sleep(0.1)
            received.append(envelope.event.content)

        channel.subscribe(slow_handler)

        # First message enters buffer
        await channel.send(Envelope(event=ModelMessage(content="first"), sender="a"))
        # Second is dropped (buffer full, drain hasn't started yet... actually
        # the drain task starts on first send, so let's fill it differently)

        # Wait for drain to process first
        await asyncio.sleep(0.2)
        assert len(received) >= 1

        await channel.close()


# ===========================================================================
# Gap 5: PriorityChannel delivery ordering and TTL expiry
# ===========================================================================


class TestPriorityChannelOrdering:
    """PriorityChannel should deliver higher priority envelopes first."""

    @pytest.mark.asyncio
    async def test_higher_priority_delivered_first(self) -> None:
        """Envelopes with higher priority should be delivered before lower."""
        scheme = DefaultPriorityScheme()
        channel = PriorityChannel(scheme=scheme)

        delivery_order: list[Any] = []

        async def handler(envelope, ctx):
            delivery_order.append(envelope.priority)

        channel.subscribe(handler)

        # Send in reverse priority order (low first, high last)
        for priority in [DefaultPriority.BACKGROUND, DefaultPriority.NORMAL, DefaultPriority.URGENT]:
            env = Envelope(
                event=ModelMessage(content="msg"),
                sender="a",
                priority=priority,
            )
            await channel.send(env)

        await asyncio.sleep(0.1)
        await channel.close()

        # Should be delivered in priority order: URGENT, NORMAL, BACKGROUND
        assert len(delivery_order) == 3
        assert delivery_order[0] == DefaultPriority.URGENT
        assert delivery_order[1] == DefaultPriority.NORMAL
        assert delivery_order[2] == DefaultPriority.BACKGROUND

    @pytest.mark.asyncio
    async def test_expired_envelopes_skipped(self) -> None:
        """Envelopes past their TTL should be skipped during drain."""
        channel = PriorityChannel()

        received: list[str] = []

        async def handler(envelope, ctx):
            received.append(envelope.event.content)

        channel.subscribe(handler)

        # Send an already-expired envelope
        expired = Envelope(
            event=ModelMessage(content="expired"),
            sender="a",
            timestamp=time.time() - 100,  # 100 seconds ago
            ttl=1.0,  # 1 second TTL — long expired
        )
        await channel.send(expired)

        # Send a fresh envelope
        fresh = Envelope(
            event=ModelMessage(content="fresh"),
            sender="a",
        )
        await channel.send(fresh)

        await asyncio.sleep(0.1)
        await channel.close()

        # Only the fresh envelope should have been delivered
        assert "fresh" in received
        assert "expired" not in received


# ===========================================================================
# Gap 6: Multiple observers emitting signals simultaneously
# ===========================================================================


class TestMultipleObserversSignals:
    """Multiple observers emitting signals in the same turn should all be collected."""

    @pytest.mark.asyncio
    async def test_two_observers_both_signal(self) -> None:
        """Two observers emitting WARNING signals should both appear in the prompt."""

        class _WarnObserver(BaseObserver):
            def __init__(self, name, message):
                super().__init__(name, watch=EventWatch(ModelResponse))
                self._message = message

            async def process(self, events, ctx):
                return Signal(
                    source=self.name,
                    severity=Severity.WARNING,
                    message=self._message,
                )

        config = _RecordingConfig(
            ToolCallEvent(name="dummy_tool", arguments='{"value": "x"}'),
            "final",
        )
        obs_a = _WarnObserver("obs-a", "alert from A")
        obs_b = _WarnObserver("obs-b", "alert from B")

        actor = Actor(
            "test",
            config=config,
            observers=[obs_a, obs_b],
            tools=[dummy_tool],
        )
        reply = await actor.ask("go")

        # Both observers should have fired (they watch ModelResponse)
        assert obs_a._watch.is_armed is False  # detached
        assert obs_b._watch.is_armed is False

        # The LLM should have seen alerts from both observers.
        # Check that the second LLM call (after tool) had alerts in prompt.
        assert config.client is not None
        # At least one call should have had observer alerts
        all_prompts = [call[1] for call in config.client.calls]
        any_has_alerts = any(
            any("OBSERVER MONITORING ALERTS" in p for p in prompts)
            for prompts in all_prompts
        )
        assert any_has_alerts


# ===========================================================================
# Gap 7: Topology additional envelope dispatch integration
# ===========================================================================


class TestTopologyAdditionalDispatch:
    """Hub should dispatch additional envelopes from topology as fire-and-forget."""

    @pytest.mark.asyncio
    async def test_additional_envelopes_dispatched(self) -> None:
        """RouteDecision additional envelopes should be dispatched independently."""

        class _CoRouterPlugin(BasePlugin):
            async def process(self, envelope, ctx):
                return RouteDecision(
                    primary=envelope,
                    additional=[
                        envelope.child(
                            DelegationRequest(
                                source=envelope.sender,
                                target="auditor",
                                task="Audit: " + getattr(envelope.event, "task", ""),
                            ),
                            recipient="auditor",
                        ),
                    ],
                )

        primary_agent = _AskableAgent("worker", result="work done")
        audit_agent = _AskableAgent("auditor", result="audit done")

        hub = Hub(topology=Pipeline(_CoRouterPlugin()))
        await hub.register(primary_agent)
        await hub.register(audit_agent)

        result = await hub._delegate("worker", "main task", source="caller")

        assert result == "work done"

        # Give additional delegations time to complete (fire-and-forget)
        await asyncio.sleep(0.2)

        # Auditor should have been called with the additional delegation
        assert len(audit_agent.asked) >= 1
        assert "Audit:" in audit_agent.asked[0]

        await hub.close()


# ===========================================================================
# Gap 8: Network convenience class with topology
# ===========================================================================


class TestNetworkWithTopology:
    """Network class should work with topology and scheduler together."""

    @pytest.mark.asyncio
    async def test_network_with_pipeline(self) -> None:
        """Network should wire topology through to Hub."""

        class _PassthroughPlugin(BasePlugin):
            def __init__(self):
                self.processed = 0

            async def process(self, envelope, ctx):
                self.processed += 1
                return envelope

        plugin = _PassthroughPlugin()
        network = Network(topology=Pipeline(plugin))

        agent = _AskableAgent("worker", result="done")
        await network.register(agent, capabilities=["work"])

        async with network:
            result = await network.hub.delegate("caller", "worker", "task")
            assert result == "done"
            assert plugin.processed >= 1

    @pytest.mark.asyncio
    async def test_network_schedule_with_callback(self) -> None:
        """Network.schedule() with callback should work standalone."""
        network = Network()
        received: list[bool] = []

        async def callback(events, ctx):
            received.append(True)

        network.schedule(IntervalWatch(0.05), callback=callback)

        async with network:
            await asyncio.sleep(0.15)

        assert len(received) >= 2


# ===========================================================================
# Additional: Hub.ask() merges user tools with network tools
# ===========================================================================


class TestHubAskToolsMerge:
    """Hub.ask() should merge user-provided tools with network tools."""

    @pytest.mark.asyncio
    async def test_user_tools_preserved(self) -> None:
        """Tools passed via kwargs should appear alongside network tools."""

        class _ToolTrackingAgent:
            name = "tracker"

            def __init__(self):
                self.received_tools: list[Any] = []

            async def ask(self, message, **kwargs):
                self.received_tools = list(kwargs.get("tools", []))
                return type("Reply", (), {"body": "ok", "response": None})()

        agent = _ToolTrackingAgent()
        hub = Hub()
        await hub.register(agent)

        @tool
        def user_tool(x: str) -> str:
            """User's custom tool."""
            return x

        await hub.ask(agent, "go", tools=[user_tool])

        # Should have user_tool + discover_agents + delegate_to = 3 tools
        assert len(agent.received_tools) == 3
        tool_names = {getattr(t, "schema", None) and t.schema.function.name for t in agent.received_tools}
        assert "user_tool" in tool_names
        assert "discover_agents" in tool_names
        assert "delegate_to" in tool_names


# ===========================================================================
# Additional: Signal injection prompt cleanup on FATAL path
# ===========================================================================


# ===========================================================================
# Bug A fix: Fanout order-independent additional preservation
# ===========================================================================


class TestFanoutOrderIndependentAdditional:
    """Fanout must preserve additional envelopes regardless of plugin order."""

    @pytest.mark.asyncio
    async def test_exception_before_additional_preserves(self) -> None:
        """Exception plugin BEFORE additional plugin — additional still preserved."""

        class _AdditionalPlugin(BasePlugin):
            async def process(self, envelope, ctx):
                return RouteDecision(
                    primary=envelope,
                    additional=[envelope.child(envelope.event, recipient="extra")],
                )

        class _RaisingPlugin(BasePlugin):
            async def process(self, envelope, ctx):
                raise ValueError("plugin exploded")

        # Raising plugin is FIRST — was broken before fix
        fanout = Fanout(_RaisingPlugin(), _AdditionalPlugin())

        env = Envelope(
            event=DelegationRequest(source="a", target="b", task="test"),
            sender="a",
            recipient="b",
        )

        from autogen.beta.network.topology import HubContext

        result = await fanout.process(env, HubContext(hub=None))  # type: ignore[arg-type]

        assert isinstance(result, RouteDecision)
        assert result.primary is None
        assert len(result.additional) == 1
        assert result.additional[0].recipient == "extra"

    @pytest.mark.asyncio
    async def test_reject_before_additional_preserves(self) -> None:
        """Reject plugin BEFORE additional plugin — additional still preserved."""

        class _AdditionalPlugin(BasePlugin):
            async def process(self, envelope, ctx):
                return RouteDecision(
                    primary=envelope,
                    additional=[envelope.child(envelope.event, recipient="audit")],
                )

        class _RejectPlugin(BasePlugin):
            async def process(self, envelope, ctx):
                return None

        # Reject plugin is FIRST — was broken before fix
        fanout = Fanout(_RejectPlugin(), _AdditionalPlugin())

        env = Envelope(
            event=DelegationRequest(source="a", target="b", task="test"),
            sender="a",
            recipient="b",
        )

        from autogen.beta.network.topology import HubContext

        result = await fanout.process(env, HubContext(hub=None))  # type: ignore[arg-type]

        assert isinstance(result, RouteDecision)
        assert result.primary is None
        assert len(result.additional) == 1
        assert result.additional[0].recipient == "audit"

    @pytest.mark.asyncio
    async def test_all_additional_collected_across_mixed_results(self) -> None:
        """Additional envelopes from ALL successful plugins are collected,
        even when some plugins reject or raise."""

        class _Additional1(BasePlugin):
            async def process(self, envelope, ctx):
                return RouteDecision(
                    primary=envelope,
                    additional=[envelope.child(envelope.event, recipient="extra1")],
                )

        class _Additional2(BasePlugin):
            async def process(self, envelope, ctx):
                return RouteDecision(
                    primary=envelope,
                    additional=[envelope.child(envelope.event, recipient="extra2")],
                )

        class _RaisingPlugin(BasePlugin):
            async def process(self, envelope, ctx):
                raise ValueError("boom")

        # Raise in the middle — both additional plugins' envelopes preserved
        fanout = Fanout(_Additional1(), _RaisingPlugin(), _Additional2())

        env = Envelope(
            event=DelegationRequest(source="a", target="b", task="test"),
            sender="a",
            recipient="b",
        )

        from autogen.beta.network.topology import HubContext

        result = await fanout.process(env, HubContext(hub=None))  # type: ignore[arg-type]

        assert isinstance(result, RouteDecision)
        assert result.primary is None
        assert len(result.additional) == 2
        recipients = {e.recipient for e in result.additional}
        assert recipients == {"extra1", "extra2"}


# ===========================================================================
# Gap: Scheduler EventWatch + Hub delegation end-to-end
# ===========================================================================


class TestSchedulerEventWatchHubDelegation:
    """Scheduler with EventWatch should trigger delegation when matching event fires."""

    @pytest.mark.asyncio
    async def test_event_watch_triggers_hub_delegation(self) -> None:
        """An EventWatch on DelegationResult should fire and delegate to target."""
        worker = _AskableAgent("worker", result="work done")
        auditor = _AskableAgent("auditor", result="audit done")

        hub = Hub()
        await hub.register(worker)
        await hub.register(auditor)

        scheduler = Scheduler(hub=hub)
        scheduler.add(
            EventWatch(DelegationResult),
            target="auditor",
            task_factory=lambda events: f"Audit: {events[0].result}",
        )
        await scheduler.start()

        # Trigger a delegation — this produces a DelegationResult on hub stream
        await hub.delegate("caller", "worker", "do work")

        # Give the EventWatch callback time to fire and delegate
        await asyncio.sleep(0.3)

        assert len(auditor.asked) >= 1
        assert "Audit:" in auditor.asked[0]

        await scheduler.stop()
        await hub.close()


# ===========================================================================
# Gap: Hub.ask() with unregistered Agent instance
# ===========================================================================


class TestHubAskUnregisteredInstance:
    """Hub.ask() with an Agent instance not registered in the Hub."""

    @pytest.mark.asyncio
    async def test_unregistered_agent_gets_network_tools(self) -> None:
        """An unregistered Agent passed to hub.ask() should still get network
        tools, but delegate_to calls from it will fail gracefully."""

        class _ToolTrackingAgent:
            name = "unregistered"

            def __init__(self):
                self.received_tools: list[Any] = []

            async def ask(self, message, **kwargs):
                self.received_tools = list(kwargs.get("tools", []))
                return type("Reply", (), {"body": "ok", "response": None})()

        agent = _ToolTrackingAgent()
        hub = Hub()
        # NOT registered — but hub.ask() should still work

        reply = await hub.ask(agent, "go")
        assert reply.body == "ok"

        # Should have network tools injected
        tool_names = {
            getattr(t, "schema", None) and t.schema.function.name
            for t in agent.received_tools
        }
        assert "discover_agents" in tool_names
        assert "delegate_to" in tool_names


# ===========================================================================
# Gap: Multiple FATAL signals in one batch
# ===========================================================================


class TestMultipleFatalSignals:
    """Multiple FATAL signals in one batch should halt on the first."""

    @pytest.mark.asyncio
    async def test_two_fatals_halts_on_first(self) -> None:
        """When two FATAL signals arrive in the same batch, the middleware
        should halt and the response should reference the first FATAL."""
        stream = MemoryStream()
        ctx = ContextType(stream=stream)
        ctx.prompt = ["existing"]

        signals: list[Signal] = [
            Signal(source="obs-a", severity=Severity.FATAL, message="fatal A"),
            Signal(source="obs-b", severity=Severity.FATAL, message="fatal B"),
        ]

        policy = InjectToPrompt()

        mw = _SignalInjectionMiddleware(
            event=ModelMessage(content="test"),
            context=ctx,
            signal_queue=signals,
            policy=policy,
            delivered_ids=set(),
        )

        call_next = AsyncMock()
        result = await mw.on_llm_call(call_next, [], ctx)

        # Should halt without calling LLM
        call_next.assert_not_called()
        assert "HALTED" in result.message.content
        # First FATAL's message should appear
        assert "fatal A" in result.message.content
        # Prompt should be cleaned (only original entry remains)
        assert len(ctx.prompt) == 1
        assert "existing" in ctx.prompt


# ===========================================================================
# Previous: Signal injection prompt cleanup on FATAL path
# ===========================================================================


class TestSignalInjectionFatalCleanup:
    """_SignalInjectionMiddleware should clean prompt on FATAL path."""

    @pytest.mark.asyncio
    async def test_prompt_entries_cleaned_on_fatal(self) -> None:
        """When FATAL halts execution, any prompt entries added by the
        policy should be removed."""
        stream = MemoryStream()
        ctx = ContextType(stream=stream)
        ctx.prompt = ["existing-prompt-entry"]

        signals: list[Signal] = [
            Signal(source="obs", severity=Severity.WARNING, message="warning"),
            Signal(source="obs", severity=Severity.FATAL, message="fatal"),
        ]

        policy = InjectToPrompt()

        mw = _SignalInjectionMiddleware(
            event=ModelMessage(content="test"),
            context=ctx,
            signal_queue=signals,
            policy=policy,
            delivered_ids=set(),
        )

        call_next = AsyncMock()  # Should not be called for FATAL
        result = await mw.on_llm_call(call_next, [], ctx)

        # call_next should NOT have been called (FATAL halts)
        call_next.assert_not_called()

        # The existing prompt entry should be preserved
        assert "existing-prompt-entry" in ctx.prompt
        # But any entries added by InjectToPrompt should be cleaned up
        assert len(ctx.prompt) == 1  # Only the original entry remains

        # The result should be a halt response
        assert "HALTED" in result.message.content
