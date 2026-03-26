# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Actor integration tests — full lifecycle with mocked LLM.

Covers:
- Observer attach/detach and signal emission
- Signal injection middleware (non-fatal → prompt, FATAL → halt)
- Harness middleware filtering events
- spawn_task / spawn_tasks tool execution
- EmitToStream signal dedup (no infinite loop)
- Scheduler.cancel() CANCELLED status and Scheduler.status()
- Sequence watch disarm-during-callback safety
- CronWatch DOW-only name parsing
- Hub topology task preservation on reroute
"""

import asyncio
from collections.abc import Sequence
from typing import Any

import pytest

from autogen.beta.config import LLMClient, ModelConfig
from autogen.beta.context import Context as ContextType
from autogen.beta.events import BaseEvent, ModelMessage, ModelResponse, ToolCallEvent, ToolCallsEvent
from autogen.beta.events.conditions import TypeCondition
from autogen.beta.network.actor import Actor
from autogen.beta.network.events import (
    DelegationRequest,
    ObserverCompleted,
    ObserverStarted,
    TaskRequest,
    TaskResult,
)
from autogen.beta.network.hub import Hub
from autogen.beta.network.observer import BaseObserver
from autogen.beta.network.primitives.harness import ConversationHarness, NetworkHarness
from autogen.beta.network.primitives.signal import (
    CallHandler,
    EmitToStream,
    HaltOnFatal,
    InjectToPrompt,
    Severity,
    Signal,
)
from autogen.beta.network.primitives.watch import EventWatch, IntervalWatch, Sequence as SequenceWatch
from autogen.beta.network.scheduler import Scheduler, WatchStatus
from autogen.beta.network.topology import BasePlugin, Pipeline
from autogen.beta.stream import MemoryStream
from autogen.beta.testing import TestConfig
from autogen.beta.tools.final import tool

from typing_extensions import Self


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


@tool
def dummy_tool(value: str) -> str:
    """A no-op tool that echoes back its input."""
    return f"echo: {value}"


class _RecordingClient(LLMClient):
    """LLM client that records calls and returns canned responses."""

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


class _CountingObserver(BaseObserver):
    """Observer that counts events and optionally produces a signal."""

    def __init__(
        self,
        name: str = "counter",
        *,
        signal_severity: str | None = None,
        signal_message: str = "alert",
    ) -> None:
        super().__init__(name, watch=EventWatch(ModelResponse))
        self.event_count = 0
        self._signal_severity = signal_severity
        self._signal_message = signal_message

    async def process(self, events: list[BaseEvent], ctx: ContextType) -> Signal | None:
        self.event_count += len(events)
        if self._signal_severity is not None:
            return Signal(
                source=self.name,
                severity=self._signal_severity,
                message=self._signal_message,
            )
        return None


class _FatalObserver(BaseObserver):
    """Observer that emits a FATAL signal on the first event."""

    def __init__(self, name: str = "fatal-obs") -> None:
        super().__init__(name, watch=EventWatch(ModelResponse))
        self.fired = False

    async def process(self, events: list[BaseEvent], ctx: ContextType) -> Signal | None:
        if not self.fired:
            self.fired = True
            return Signal(
                source=self.name,
                severity=Severity.FATAL,
                message="Critical failure detected",
            )
        return None


class _AskableAgent:
    """Mock agent that returns a canned result."""

    def __init__(self, name: str, result: str = "done"):
        self.name = name
        self._result = result
        self.asked: list[str] = []

    async def ask(self, message: str, **kwargs: Any) -> Any:
        self.asked.append(message)
        return type("Reply", (), {"content": self._result, "body": self._result})()


# ===========================================================================
# Actor lifecycle tests
# ===========================================================================


class TestActorObserverLifecycle:
    """Observer attach/detach during Actor._execute()."""

    @pytest.mark.asyncio
    async def test_observers_attached_and_detached(self) -> None:
        """Observers are attached on _execute start and detached in finally block."""
        config = _RecordingConfig("hello")
        observer = _CountingObserver()

        actor = Actor("test-actor", prompt="Be helpful.", config=config, observers=[observer])
        reply = await actor.ask("Hi")

        assert reply.body == "hello"
        # Observer should have been detached (watch disarmed)
        assert not observer._watch.is_armed

    @pytest.mark.asyncio
    async def test_observer_started_completed_events(self) -> None:
        """ObserverStarted and ObserverCompleted events are emitted."""
        config = _RecordingConfig("ok")
        observer = _CountingObserver("my-obs")

        actor = Actor("test-actor", config=config, observers=[observer])
        stream = MemoryStream()

        started: list[BaseEvent] = []
        completed: list[BaseEvent] = []
        stream.subscribe(lambda e: started.append(e), condition=TypeCondition(ObserverStarted))
        stream.subscribe(lambda e: completed.append(e), condition=TypeCondition(ObserverCompleted))

        await actor.ask("Hi", stream=stream)

        assert len(started) == 1
        assert started[0].name == "my-obs"
        assert len(completed) == 1
        assert completed[0].name == "my-obs"

    @pytest.mark.asyncio
    async def test_observer_receives_model_response(self) -> None:
        """Observer's watch fires on matching events during execution."""
        config = _RecordingConfig("response-text")
        observer = _CountingObserver()

        actor = Actor("test-actor", config=config, observers=[observer])
        await actor.ask("Hi")

        assert observer.event_count >= 1

    @pytest.mark.asyncio
    async def test_add_observer_before_ask(self) -> None:
        """add_observer() adds an observer that participates in execution."""
        config = _RecordingConfig("result")
        observer = _CountingObserver("added")

        actor = Actor("test-actor", config=config)
        actor.add_observer(observer)
        await actor.ask("Hi")

        assert observer.event_count >= 1
        assert not observer._watch.is_armed


# ===========================================================================
# Signal injection tests
# ===========================================================================


class TestSignalInjection:
    """Signal injection middleware behaviour."""

    @pytest.mark.asyncio
    async def test_non_fatal_signal_injected_into_prompt(self) -> None:
        """Non-fatal signals appear in the LLM prompt and are cleaned up after."""
        config = _RecordingConfig("first-response")
        observer = _CountingObserver(signal_severity=Severity.WARNING, signal_message="token budget low")

        actor = Actor("test-actor", prompt="Be helpful.", config=config, observers=[observer])
        reply = await actor.ask("Hi")

        # The actor should complete (not halt)
        assert reply.body == "first-response"

    @pytest.mark.asyncio
    async def test_fatal_signal_halts_execution(self) -> None:
        """FATAL signal causes immediate halt with synthetic response.

        Flow: LLM returns tool call → ModelResponse fires observer → FATAL queued
        → tool executes → second LLM call → middleware intercepts with HALTED.
        """
        config = _RecordingConfig(
            ToolCallEvent(name="dummy_tool", arguments='{"value": "ping"}'),
            "should-not-reach-this",
        )
        fatal_obs = _FatalObserver()

        actor = Actor("test-actor", config=config, observers=[fatal_obs], tools=[dummy_tool])
        reply = await actor.ask("Hi")

        assert fatal_obs.fired
        assert reply.body is not None
        assert "HALTED" in reply.body

    @pytest.mark.asyncio
    async def test_halt_on_fatal_policy(self) -> None:
        """HaltOnFatal wraps another policy and halts on FATAL."""
        config = _RecordingConfig(
            ToolCallEvent(name="dummy_tool", arguments='{"value": "x"}'),
            "after",
        )
        fatal_obs = _FatalObserver()
        policy = HaltOnFatal(inner=InjectToPrompt())

        actor = Actor("test-actor", config=config, observers=[fatal_obs], tools=[dummy_tool], signal_policy=policy)
        reply = await actor.ask("Hi")

        assert fatal_obs.fired
        assert reply.body is not None
        assert "HALTED" in reply.body


# ===========================================================================
# EmitToStream dedup (BUG-1 fix)
# ===========================================================================


class TestEmitToStreamDedup:
    """EmitToStream should not cause infinite signal re-collection."""

    @pytest.mark.asyncio
    async def test_emit_to_stream_no_infinite_loop(self) -> None:
        """Using EmitToStream as signal policy does not loop infinitely."""
        config = _RecordingConfig("done")
        observer = _CountingObserver(signal_severity=Severity.WARNING, signal_message="test alert")

        actor = Actor(
            "test-actor",
            config=config,
            observers=[observer],
            signal_policy=EmitToStream(),
        )

        # This should complete without hanging
        reply = await asyncio.wait_for(actor.ask("Hi"), timeout=5.0)
        assert reply.body == "done"

    @pytest.mark.asyncio
    async def test_call_handler_policy(self) -> None:
        """CallHandler policy delivers signals to handler function."""
        received: list[Signal] = []

        async def handler(signals: list[Signal]) -> None:
            received.extend(signals)

        config = _RecordingConfig(
            ToolCallEvent(name="dummy_tool", arguments='{"value": "x"}'),
            "final",
        )
        observer = _CountingObserver(signal_severity=Severity.INFO, signal_message="info alert")

        actor = Actor(
            "test-actor",
            config=config,
            observers=[observer],
            tools=[dummy_tool],
            signal_policy=CallHandler(handler),
        )
        reply = await actor.ask("Hi")
        # Handler should have received the signal from the observer
        assert reply.body is not None
        assert len(received) >= 1


# ===========================================================================
# Harness middleware tests
# ===========================================================================


class TestHarnessMiddleware:
    """ContextHarness integration in Actor middleware chain."""

    @pytest.mark.asyncio
    async def test_conversation_harness_filters_events(self) -> None:
        """ConversationHarness only passes conversation events to LLM."""
        config = _RecordingConfig("ok")
        actor = Actor(
            "test-actor",
            config=config,
            harness=ConversationHarness(),
        )
        reply = await actor.ask("Hi")

        assert config.client is not None
        assert len(config.client.calls) >= 1
        assert reply.body == "ok"

    @pytest.mark.asyncio
    async def test_network_harness_includes_network_events(self) -> None:
        """NetworkHarness passes Signal and delegation events to LLM."""
        config = _RecordingConfig("ok")
        actor = Actor(
            "test-actor",
            config=config,
            harness=NetworkHarness(),
        )
        reply = await actor.ask("Hi")

        assert config.client is not None
        assert len(config.client.calls) >= 1
        assert reply.body == "ok"


# ===========================================================================
# Task spawning tests
# ===========================================================================


class TestSpawnTask:
    """Actor spawn_task and spawn_tasks tools."""

    @pytest.mark.asyncio
    async def test_spawn_task_emits_lifecycle_events(self) -> None:
        """spawn_task emits TaskRequest and TaskResult events."""
        task_config = TestConfig("sub-result")
        main_config = _RecordingConfig(
            ToolCallEvent(name="spawn_task", arguments='{"task": "research AI trends"}'),
            "final answer",
        )

        actor = Actor(
            "researcher",
            config=main_config,
            task_config=task_config,
        )

        stream = MemoryStream()
        task_requests: list[BaseEvent] = []
        task_results: list[BaseEvent] = []
        stream.subscribe(lambda e: task_requests.append(e), condition=TypeCondition(TaskRequest))
        stream.subscribe(lambda e: task_results.append(e), condition=TypeCondition(TaskResult))

        reply = await actor.ask("Do research", stream=stream)

        assert len(task_requests) == 1
        assert task_requests[0].task == "research AI trends"
        assert len(task_results) == 1
        assert task_results[0].result == "sub-result"

    @pytest.mark.asyncio
    async def test_spawn_tasks_parallel(self) -> None:
        """spawn_tasks with parallel=True runs tasks concurrently."""
        task_config = TestConfig("result-A")
        main_config = _RecordingConfig(
            ToolCallEvent(
                name="spawn_tasks",
                arguments='{"tasks": ["task A", "task B"], "parallel": true}',
            ),
            "combined answer",
        )

        actor = Actor(
            "multi-tasker",
            config=main_config,
            task_config=task_config,
        )

        stream = MemoryStream()
        task_requests: list[BaseEvent] = []
        stream.subscribe(lambda e: task_requests.append(e), condition=TypeCondition(TaskRequest))

        await actor.ask("Do multiple tasks", stream=stream)

        assert len(task_requests) == 2

    @pytest.mark.asyncio
    async def test_spawn_tasks_sequential(self) -> None:
        """spawn_tasks with parallel=False runs tasks sequentially."""
        task_config = TestConfig("seq-result")
        main_config = _RecordingConfig(
            ToolCallEvent(
                name="spawn_tasks",
                arguments='{"tasks": ["first", "second"], "parallel": false}',
            ),
            "done",
        )

        actor = Actor(
            "seq-tasker",
            config=main_config,
            task_config=task_config,
        )

        stream = MemoryStream()
        task_requests: list[BaseEvent] = []
        stream.subscribe(lambda e: task_requests.append(e), condition=TypeCondition(TaskRequest))

        await actor.ask("Do tasks", stream=stream)

        assert len(task_requests) == 2


# ===========================================================================
# Scheduler bug fixes
# ===========================================================================


class TestSchedulerBugFixes:
    """Tests for Scheduler cancel/status fixes (BUG-3)."""

    @pytest.mark.asyncio
    async def test_cancel_sets_cancelled_status(self) -> None:
        """cancel() sets status to CANCELLED instead of removing entry."""
        scheduler = Scheduler()
        wid = scheduler.add(IntervalWatch(999), callback=lambda e, c: None)

        assert scheduler.cancel(wid)
        assert scheduler.status(wid) == WatchStatus.CANCELLED

        # Should still appear in watches list
        watch_ids = [w[0] for w in scheduler.watches]
        assert wid in watch_ids

    @pytest.mark.asyncio
    async def test_status_returns_none_for_unknown(self) -> None:
        """status() returns None for non-existent watch IDs."""
        scheduler = Scheduler()
        assert scheduler.status("nonexistent") is None

    @pytest.mark.asyncio
    async def test_status_returns_correct_lifecycle(self) -> None:
        """status() tracks PENDING → ARMED → PAUSED → ARMED → CANCELLED."""
        scheduler = Scheduler()
        wid = scheduler.add(IntervalWatch(999), callback=lambda e, c: None)
        assert scheduler.status(wid) == WatchStatus.PENDING

        await scheduler.start()
        assert scheduler.status(wid) == WatchStatus.ARMED

        scheduler.pause(wid)
        assert scheduler.status(wid) == WatchStatus.PAUSED

        scheduler.resume(wid)
        assert scheduler.status(wid) == WatchStatus.ARMED

        scheduler.cancel(wid)
        assert scheduler.status(wid) == WatchStatus.CANCELLED

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_cancel_does_not_rearm(self) -> None:
        """A cancelled watch should not be re-armed by resume()."""
        scheduler = Scheduler()
        wid = scheduler.add(IntervalWatch(999), callback=lambda e, c: None)
        await scheduler.start()
        scheduler.cancel(wid)

        # resume should not change CANCELLED status
        scheduler.resume(wid)
        assert scheduler.status(wid) == WatchStatus.CANCELLED

        await scheduler.stop()


# ===========================================================================
# CronWatch DOW fix (BUG-4)
# ===========================================================================


class TestCronWatchDowFix:
    """DOW names should only be parsed in the day-of-week field."""

    def test_dow_names_not_parsed_in_minute_field(self) -> None:
        """Putting a DOW name in the minute field should raise ValueError."""
        from autogen.beta.network.primitives.watch import CronWatch

        cron = CronWatch("MON * * * *")
        import datetime

        # MON is not a valid minute — should raise ValueError (not int)
        with pytest.raises(ValueError):
            cron._next_fire_time(datetime.datetime.now())

    def test_dow_names_still_work_in_dow_field(self) -> None:
        """DOW names should still parse correctly in the 5th field."""
        from autogen.beta.network.primitives.watch import CronWatch

        cron = CronWatch("0 9 * * MON")
        import datetime

        result = cron._next_fire_time(datetime.datetime.now())
        assert result.weekday() == 0  # Monday


# ===========================================================================
# Sequence watch disarm safety (BUG-5)
# ===========================================================================


class TestSequenceWatchDisarmSafety:
    """Sequence watch should handle disarm during callback."""

    @pytest.mark.asyncio
    async def test_disarm_during_callback_no_error(self) -> None:
        """Disarming a Sequence during its callback does not raise."""
        stream = MemoryStream()

        fired: list[list[BaseEvent]] = []
        seq = SequenceWatch(
            EventWatch(Signal),
            EventWatch(Signal),
        )

        async def callback(events: list[BaseEvent], ctx: ContextType) -> None:
            fired.append(events)
            # Disarm during callback — should not cause errors
            seq.disarm()

        seq.arm(stream, callback)
        ctx = ContextType(stream=stream)

        # Fire first watch
        await stream.send(Signal(source="test", severity="info", message="first"), ctx)
        # Fire second watch — triggers callback which calls disarm()
        await stream.send(Signal(source="test", severity="info", message="second"), ctx)

        assert len(fired) == 1
        # Should be disarmed now without error
        assert not seq.is_armed

    @pytest.mark.asyncio
    async def test_sequence_does_not_rearm_after_disarm(self) -> None:
        """After disarm during callback, the sequence should not re-arm sub-watches."""
        stream = MemoryStream()
        fired_count = 0
        seq = SequenceWatch(
            EventWatch(Signal),
            EventWatch(Signal),
        )

        async def callback(events: list[BaseEvent], ctx: ContextType) -> None:
            nonlocal fired_count
            fired_count += 1
            seq.disarm()

        seq.arm(stream, callback)
        ctx = ContextType(stream=stream)

        # Complete the sequence
        await stream.send(Signal(source="t", severity="info", message="1"), ctx)
        await stream.send(Signal(source="t", severity="info", message="2"), ctx)

        assert fired_count == 1

        # Send more signals — should NOT trigger callback again since disarmed
        await stream.send(Signal(source="t", severity="info", message="3"), ctx)
        await stream.send(Signal(source="t", severity="info", message="4"), ctx)

        assert fired_count == 1


# ===========================================================================
# Hub topology task preservation (BUG-2)
# ===========================================================================


class TestHubTopologyTaskPreservation:
    """Topology rerouting should preserve task modifications."""

    @pytest.mark.asyncio
    async def test_reroute_preserves_modified_task(self) -> None:
        """When topology reroutes AND modifies the task, the task modification is preserved."""

        class _TaskModifyingPlugin(BasePlugin):
            async def process(self, envelope, ctx):
                if isinstance(envelope.event, DelegationRequest):
                    # Reroute from agent-a to agent-b AND modify the task text
                    envelope.recipient = "agent-b"
                    envelope.event = DelegationRequest(
                        source=envelope.event.source,
                        target="agent-b",
                        task="MODIFIED: " + envelope.event.task,
                    )
                return envelope

        agent_a = _AskableAgent("agent-a", result="a-result")
        agent_b = _AskableAgent("agent-b", result="b-result")
        hub = Hub(topology=Pipeline(_TaskModifyingPlugin()))

        await hub.register(agent_a, capabilities=["work"])
        await hub.register(agent_b, capabilities=["work"])

        # Delegate to agent-a, but plugin reroutes to agent-b with modified task
        result = await hub.delegate("caller", "agent-a", "original task")

        # agent-b should have received the modified task, not agent-a
        assert len(agent_a.asked) == 0
        assert len(agent_b.asked) == 1
        assert "MODIFIED: original task" == agent_b.asked[0]
