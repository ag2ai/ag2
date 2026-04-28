# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Observer smoke: StreamObserver, BaseObserver (Watch-based), TokenMonitor,
LoopDetector, ObserverAlert → HaltEvent short-circuit, ObserverStarted/Completed.

All real LLM calls via Gemini 3 Flash Preview.
"""

import pytest

from autogen.beta import Agent, observer
from autogen.beta.events import (
    BaseEvent,
    ModelMessageChunk,
    ModelResponse,
)
from autogen.beta.events.alert import HaltEvent, ObserverAlert, Severity
from autogen.beta.events.lifecycle import ObserverCompleted, ObserverStarted
from autogen.beta.observer import BaseObserver, LoopDetector, TokenMonitor
from autogen.beta.policies import AlertPolicy
from autogen.beta.stream import MemoryStream
from autogen.beta.watch import EventWatch

pytestmark = pytest.mark.asyncio


async def test_stream_observer_decorator_sees_responses(provider_config) -> None:
    """@observer(ModelResponse, cb) captures every ModelResponse event."""
    seen: list[ModelResponse] = []

    def on_response(event: ModelResponse) -> None:
        seen.append(event)

    obs = observer(ModelResponse, on_response)

    agent = Agent("watcher", config=provider_config, observers=[obs])
    await agent.ask("Say 'hi'.")

    assert len(seen) >= 1
    assert seen[0].content is not None


async def test_observer_sees_streamed_chunks(streaming_config) -> None:
    """@observer can subscribe to transient chunk events.

    Uses streaming-enabled config and asserts the observer actually received
    chunks and that they reconstruct the final body.
    """
    chunks: list[str] = []

    def on_chunk(event: ModelMessageChunk) -> None:
        chunks.append(event.content)

    obs = observer(ModelMessageChunk, on_chunk)

    agent = Agent("chunker", config=streaming_config, observers=[obs])
    reply = await agent.ask("Say the single word 'ocean'.")
    assert reply.body is not None
    assert chunks, "streaming observer must receive at least one chunk"
    assert "".join(chunks) == reply.body


async def test_base_observer_event_watch_fires(provider_config) -> None:
    """BaseObserver with EventWatch(ModelResponse) — process runs per response."""
    processed: list[int] = []

    class CountingObserver(BaseObserver):
        async def process(self, events: list[BaseEvent], ctx) -> ObserverAlert | None:
            processed.append(len(events))
            return None

    obs = CountingObserver("counter", watch=EventWatch(ModelResponse))

    agent = Agent("base", config=provider_config, observers=[obs])
    await agent.ask("Say 'ok'.")

    assert len(processed) >= 1
    assert processed[0] >= 1


async def test_base_observer_returns_alert_on_stream(provider_config) -> None:
    """BaseObserver.process returning an ObserverAlert emits it on the stream."""
    alerts: list[ObserverAlert] = []

    class NoisyObserver(BaseObserver):
        async def process(self, events: list[BaseEvent], ctx) -> ObserverAlert | None:
            return ObserverAlert(
                source=self.name,
                severity=Severity.WARNING,
                message="a response was observed",
            )

    obs = NoisyObserver("noisy", watch=EventWatch(ModelResponse))

    stream = MemoryStream()
    stream.where(ObserverAlert).subscribe(lambda e: alerts.append(e))

    agent = Agent("alerter", config=provider_config, observers=[obs])
    await agent.ask("Say 'ok'.", stream=stream)

    assert len(alerts) >= 1
    assert alerts[0].severity == "warning"
    assert alerts[0].source == "noisy"


async def test_token_monitor_builtin(provider_config) -> None:
    """TokenMonitor tallies cumulative usage and issues alerts above threshold."""
    # Very low thresholds so the very first reply trips the warning
    monitor = TokenMonitor(warn_threshold=10, alert_threshold=100_000)

    alerts: list[ObserverAlert] = []
    stream = MemoryStream()
    stream.where(ObserverAlert).subscribe(lambda e: alerts.append(e))

    agent = Agent("tokens", config=provider_config, observers=[monitor])
    reply = await agent.ask("Write a 30-word paragraph about a river.", stream=stream)
    assert reply.body is not None

    assert monitor.total_tokens > 0
    assert any(a.source == "token-monitor" for a in alerts)


async def test_loop_detector_builtin(provider_config) -> None:
    """LoopDetector emits an ObserverAlert when a tool is called repeatedly.

    The tool returns "pending" the first 3 calls and "ready" after that,
    so the LLM exits the retry loop naturally. ``repeat_threshold=2`` is
    crossed during the pending phase, so the detector still fires.
    """
    detector = LoopDetector(window_size=10, repeat_threshold=2)

    alerts: list[ObserverAlert] = []
    stream = MemoryStream()
    stream.where(ObserverAlert).subscribe(lambda e: alerts.append(e))

    call_count = 0

    def get_status() -> str:
        """Return the current system status."""
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return "Status: pending. Call this tool again to retry."
        return "Status: ready."

    agent = Agent(
        "looper",
        prompt="Call get_status until it returns a non-pending status, then stop.",
        config=provider_config,
        tools=[get_status],
        observers=[detector],
    )
    reply = await agent.ask("Get the system status. Retry until non-pending.", stream=stream)
    assert reply.body is not None
    loop_alerts = [a for a in alerts if "loop" in a.message.lower()]
    assert loop_alerts, f"LoopDetector did not fire; got alerts={alerts!r}"
    assert loop_alerts[0].source == "loop-detector"


async def test_alert_policy_fatal_halts_llm(provider_config) -> None:
    """A FATAL ObserverAlert + AlertPolicy short-circuits the LLM call.

    Flow: observer emits FATAL → AlertPolicy routes → HaltEvent fires →
    _HaltCheckMiddleware swaps the LLM call for a synthetic HALTED reply.
    """
    halt_events: list[HaltEvent] = []

    class FatalOnFirstResponse(BaseObserver):
        """Emits a FATAL alert on the very first ModelResponse."""

        def __init__(self) -> None:
            super().__init__("fatal-obs", watch=EventWatch(ModelResponse))
            self._fired = False

        async def process(self, events: list[BaseEvent], ctx) -> ObserverAlert | None:
            if self._fired:
                return None
            self._fired = True
            return ObserverAlert(
                source=self.name,
                severity=Severity.FATAL,
                message="hard stop",
            )

    stream = MemoryStream()
    stream.where(HaltEvent).subscribe(lambda e: halt_events.append(e))

    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    agent = Agent(
        "halter",
        prompt="Use the add tool to compute 5+5, then say done.",
        config=provider_config,
        tools=[add],
        observers=[FatalOnFirstResponse()],
        assembly=[AlertPolicy()],
    )

    # First turn produces a response → observer fires FATAL → next LLM call halted.
    # We need at least one inner tool-call loop to hit the halt check.
    reply = await agent.ask("Compute 5+5 using the add tool, then say 'done'.", stream=stream)

    # Either the reply was halted (contains HALTED marker) or we observed a HaltEvent
    assert reply.body is not None
    assert len(halt_events) >= 1 or "HALTED" in (reply.body or "")


async def test_observer_lifecycle_events_emitted(provider_config) -> None:
    """ObserverStarted / ObserverCompleted fire around Agent execution."""
    started: list[ObserverStarted] = []
    completed: list[ObserverCompleted] = []

    stream = MemoryStream()
    stream.where(ObserverStarted).subscribe(lambda e: started.append(e))
    stream.where(ObserverCompleted).subscribe(lambda e: completed.append(e))

    def noop(event: ModelResponse) -> None:
        return None

    obs_a = observer(ModelResponse, noop)
    obs_b = observer(ModelResponse, noop)

    # These both end up unnamed — the agent uses type(obs).__name__ = StreamObserver
    agent = Agent("lifecycle", config=provider_config, observers=[obs_a, obs_b])
    await agent.ask("Say 'ok'.", stream=stream)

    # Two observers → two Started, two Completed
    assert len(started) == 2
    assert len(completed) == 2


async def test_per_ask_observer_augments(provider_config) -> None:
    """Observers passed to .ask() are added for that turn only."""
    seen_at_ask: list[str] = []

    def on_resp(event: ModelResponse) -> None:
        seen_at_ask.append(event.content or "")

    agent = Agent("per-ask-obs", config=provider_config)
    obs = observer(ModelResponse, on_resp)
    await agent.ask("Say 'hello'.", observers=[obs])

    assert len(seen_at_ask) >= 1
