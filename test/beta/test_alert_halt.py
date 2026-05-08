# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for AlertPolicy and _HaltCheckMiddleware.

AlertPolicy is the assembly-chain replacement for the old Signal delivery
system. _HaltCheckMiddleware catches HaltEvent and short-circuits LLM calls.

These tests isolate both components from the full Agent stack to verify
edge cases not covered by Agent integration tests.
"""

from collections.abc import Sequence
from typing import Any
from unittest.mock import MagicMock

import pytest
from typing_extensions import Self

from autogen.beta import Agent
from autogen.beta.config import LLMClient, ModelConfig
from autogen.beta.context import ConversationContext as ContextType
from autogen.beta.events import BaseEvent, ModelMessage, ModelResponse, ToolCallEvent, ToolCallsEvent
from autogen.beta.events.alert import HaltEvent, ObserverAlert, Severity
from autogen.beta.observer import BaseObserver
from autogen.beta.policies import AlertPolicy
from autogen.beta.stream import MemoryStream
from autogen.beta.tools.final import tool
from autogen.beta.watch import EventWatch


@tool
def echo_tool(value: str) -> str:
    """Echoes back input."""
    return f"echo: {value}"


class _RecordingClient(LLMClient):
    """LLM client that records calls and returns canned responses."""

    def __init__(self, *responses: ModelResponse | ToolCallEvent | str) -> None:
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
        resp = self._responses[self._call_count] if self._call_count < len(self._responses) else "done"
        self._call_count += 1
        if isinstance(resp, str):
            return ModelResponse(message=ModelMessage(content=resp))
        if isinstance(resp, ToolCallEvent):
            return ModelResponse(tool_calls=ToolCallsEvent(calls=[resp]))
        return resp


class _RecordingConfig(ModelConfig):
    __test__ = False

    def __init__(self, *responses: ModelResponse | ToolCallEvent | str) -> None:
        self._responses = responses
        self.client: _RecordingClient | None = None

    def copy(self) -> Self:
        return self

    def create(self) -> _RecordingClient:
        self.client = _RecordingClient(*self._responses)
        return self.client


class TestAlertPolicyUnit:
    """AlertPolicy in isolation (no Agent, no middleware)."""

    @pytest.mark.asyncio
    async def test_no_alerts_is_noop(self) -> None:
        """When no ObserverAlert events are present, policy is a no-op."""
        policy = AlertPolicy()
        ctx = MagicMock()

        prompts = ["System prompt"]
        events: list[BaseEvent] = [ModelMessage(content="hello")]

        result_prompts, result_events = await policy.apply(prompts, events, ctx)

        assert result_prompts == ["System prompt"]
        assert result_events == events
        ctx.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_single_warning_injected(self) -> None:
        """A single non-fatal alert is injected as prompt text."""
        policy = AlertPolicy()
        ctx = MagicMock()

        alert = ObserverAlert(source="monitor", severity=Severity.WARNING, message="Token budget low")
        prompts = ["System prompt"]
        events: list[BaseEvent] = [ModelMessage(content="hi"), alert]

        result_prompts, _ = await policy.apply(prompts, events, ctx)

        assert len(result_prompts) == 2
        assert "[OBSERVER MONITORING ALERTS]" in result_prompts[1]
        assert "Token budget low" in result_prompts[1]
        assert "WARNING" in result_prompts[1]
        ctx.send.assert_not_called()  # No HaltEvent for non-fatal

    @pytest.mark.asyncio
    async def test_multiple_severities_formatted(self) -> None:
        """Multiple alerts with different severities are all included."""
        policy = AlertPolicy()
        ctx = MagicMock()

        alerts = [
            ObserverAlert(source="a", severity=Severity.INFO, message="Info msg"),
            ObserverAlert(source="b", severity=Severity.WARNING, message="Warn msg"),
            ObserverAlert(source="c", severity=Severity.CRITICAL, message="Critical msg"),
        ]
        prompts = ["System"]
        events: list[BaseEvent] = list(alerts)

        result_prompts, _ = await policy.apply(prompts, events, ctx)

        alert_text = result_prompts[1]
        assert "Info msg" in alert_text
        assert "Warn msg" in alert_text
        assert "Critical msg" in alert_text

    @pytest.mark.asyncio
    async def test_fatal_emits_halt_event(self) -> None:
        """FATAL alert emits HaltEvent on the stream via context.send()."""
        policy = AlertPolicy()
        sent: list[BaseEvent] = []

        async def mock_send(event: BaseEvent) -> None:
            sent.append(event)

        ctx = MagicMock()
        ctx.send = mock_send

        alert = ObserverAlert(source="guard", severity=Severity.FATAL, message="Budget exceeded")
        prompts = ["System"]
        events: list[BaseEvent] = [alert]

        result_prompts, _ = await policy.apply(prompts, events, ctx)

        # HaltEvent should have been emitted
        assert len(sent) == 1
        assert isinstance(sent[0], HaltEvent)
        assert "Budget exceeded" in sent[0].reason
        assert sent[0].source == "guard"

        # Halt note should be in prompts
        assert any("[FATAL ALERT]" in p for p in result_prompts)

    @pytest.mark.asyncio
    async def test_mixed_fatal_and_nonfatal(self) -> None:
        """Both fatal and non-fatal alerts are handled correctly together."""
        policy = AlertPolicy()
        sent: list[BaseEvent] = []

        async def mock_send(event: BaseEvent) -> None:
            sent.append(event)

        ctx = MagicMock()
        ctx.send = mock_send

        alerts = [
            ObserverAlert(source="a", severity=Severity.WARNING, message="Just a warning"),
            ObserverAlert(source="b", severity=Severity.FATAL, message="Total failure"),
        ]
        prompts = ["System"]
        events: list[BaseEvent] = list(alerts)

        result_prompts, _ = await policy.apply(prompts, events, ctx)

        # Non-fatal should be injected
        assert any("Just a warning" in p for p in result_prompts)
        # Fatal should emit HaltEvent
        assert any(isinstance(e, HaltEvent) for e in sent)
        # Fatal note should be in prompts
        assert any("[FATAL ALERT]" in p for p in result_prompts)

    @pytest.mark.asyncio
    async def test_dedup_across_calls(self) -> None:
        """Same alerts are not re-injected on subsequent apply() calls."""
        policy = AlertPolicy()
        ctx = MagicMock()

        alert = ObserverAlert(source="mon", severity=Severity.INFO, message="Noted")
        prompts = ["System"]
        events: list[BaseEvent] = [alert]

        # First call — alert delivered
        result_prompts_1, _ = await policy.apply(prompts, events, ctx)
        assert len(result_prompts_1) == 2

        # Second call — same alert should be skipped
        result_prompts_2, _ = await policy.apply(prompts, events, ctx)
        assert len(result_prompts_2) == 1  # Only original prompt, no alerts

    @pytest.mark.asyncio
    async def test_new_alerts_delivered_after_dedup(self) -> None:
        """New alerts are delivered even after previous ones were deduped."""
        policy = AlertPolicy()
        ctx = MagicMock()

        alert1 = ObserverAlert(source="a", severity=Severity.INFO, message="First")
        alert2 = ObserverAlert(source="b", severity=Severity.WARNING, message="Second")

        # First call — deliver alert1
        prompts1, _ = await policy.apply(["System"], [alert1], ctx)
        assert "First" in prompts1[1]

        # Second call — alert1 deduped, alert2 delivered
        prompts2, _ = await policy.apply(["System"], [alert1, alert2], ctx)
        assert len(prompts2) == 2
        assert "Second" in prompts2[1]
        assert "First" not in prompts2[1]

    @pytest.mark.asyncio
    async def test_dedup_survives_history_replacement(self) -> None:
        """A re-constructed alert with identical content is still treated as a duplicate.

        Compaction rebuilds events from disk into fresh objects; an
        ``id(event)``-based dedup would re-deliver the alert. Content
        dedup (source, severity, message) must hold.
        """
        policy = AlertPolicy()
        ctx = MagicMock()

        original = ObserverAlert(source="mon", severity=Severity.WARNING, message="watch out")
        await policy.apply(["System"], [original], ctx)

        replaced = ObserverAlert(source="mon", severity=Severity.WARNING, message="watch out")
        assert original is not replaced

        prompts, _ = await policy.apply(["System"], [replaced], ctx)
        assert all("watch out" not in p for p in prompts)

    @pytest.mark.asyncio
    async def test_multiple_fatals_uses_first_for_halt(self) -> None:
        """When multiple FATAL alerts arrive, the first one drives the HaltEvent."""
        policy = AlertPolicy()
        sent: list[BaseEvent] = []

        async def mock_send(event: BaseEvent) -> None:
            sent.append(event)

        ctx = MagicMock()
        ctx.send = mock_send

        alerts = [
            ObserverAlert(source="obs-a", severity=Severity.FATAL, message="Crash A"),
            ObserverAlert(source="obs-b", severity=Severity.FATAL, message="Crash B"),
        ]

        await policy.apply(["System"], list(alerts), ctx)

        assert len(sent) == 1
        halt = sent[0]
        assert isinstance(halt, HaltEvent)
        assert halt.source == "obs-a"
        assert "Crash A" in halt.reason
        # Both fatals should be in the alerts list
        assert len(halt.alerts) == 2


class TestHaltCheckMiddleware:
    """_HaltCheckMiddleware short-circuits LLM on HaltEvent."""

    @pytest.mark.asyncio
    async def test_no_halt_passes_through(self) -> None:
        """Without any FATAL alert, LLM is called normally."""
        config = _RecordingConfig("normal response")

        agent = Agent("test", config=config, assembly=[AlertPolicy()])
        reply = await agent.ask("Hi")

        assert reply.body == "normal response"
        assert config.client is not None
        assert config.client._call_count == 1

    @pytest.mark.asyncio
    async def test_halt_short_circuits_second_llm_call(self) -> None:
        """FATAL alert after first LLM call prevents second call.

        Flow: LLM call 1 returns tool call -> ModelResponse fires FATAL observer
        -> tool executes -> LLM call 2 intercepted by _HaltCheckMiddleware.
        """
        config = _RecordingConfig(
            ToolCallEvent(name="echo_tool", arguments='{"value": "test"}'),
            "should-not-reach",
        )

        class _FatalOnFirst(BaseObserver):
            def __init__(self) -> None:
                super().__init__("fatal-first", watch=EventWatch(ModelResponse))
                self.fired = False

            async def process(self, events: list[BaseEvent], ctx: ContextType) -> ObserverAlert | None:
                if not self.fired:
                    self.fired = True
                    return ObserverAlert(
                        source="fatal-first",
                        severity=Severity.FATAL,
                        message="Stop now",
                    )
                return None

        observer = _FatalOnFirst()
        agent = Agent(
            "test",
            config=config,
            observers=[observer],
            tools=[echo_tool],
            assembly=[AlertPolicy()],
        )
        reply = await agent.ask("Hi")

        assert observer.fired
        assert reply.body is not None
        assert "HALTED" in reply.body
        # Second LLM call was short-circuited
        assert config.client is not None
        assert config.client._call_count == 1

    @pytest.mark.asyncio
    async def test_halt_event_on_stream(self) -> None:
        """HaltEvent is observable on the stream."""
        config = _RecordingConfig(
            ToolCallEvent(name="echo_tool", arguments='{"value": "x"}'),
            "after",
        )

        class _Fatal(BaseObserver):
            def __init__(self) -> None:
                super().__init__("fatal", watch=EventWatch(ModelResponse))

            async def process(self, events: list[BaseEvent], ctx: ContextType) -> ObserverAlert | None:
                return ObserverAlert(
                    source="fatal",
                    severity=Severity.FATAL,
                    message="halt",
                )

        stream = MemoryStream()
        halts: list[HaltEvent] = []
        stream.subscribe(lambda e: halts.append(e))

        agent = Agent(
            "test",
            config=config,
            observers=[_Fatal()],
            tools=[echo_tool],
            assembly=[AlertPolicy()],
        )
        await agent.ask("Hi", stream=stream)

        halt_events = [e for e in halts if isinstance(e, HaltEvent)]
        assert len(halt_events) >= 1
        assert "halt" in halt_events[0].reason

    @pytest.mark.asyncio
    async def test_nonfatal_does_not_halt(self) -> None:
        """Non-fatal alerts don't trigger _HaltCheckMiddleware."""
        config = _RecordingConfig(
            ToolCallEvent(name="echo_tool", arguments='{"value": "x"}'),
            "final result",
        )

        class _Warning(BaseObserver):
            def __init__(self) -> None:
                super().__init__("warner", watch=EventWatch(ModelResponse))

            async def process(self, events: list[BaseEvent], ctx: ContextType) -> ObserverAlert | None:
                return ObserverAlert(
                    source="warner",
                    severity=Severity.WARNING,
                    message="be careful",
                )

        agent = Agent(
            "test",
            config=config,
            observers=[_Warning()],
            tools=[echo_tool],
            assembly=[AlertPolicy()],
        )
        reply = await agent.ask("Hi")

        # Should complete normally — second LLM call happens
        assert reply.body == "final result"
        assert config.client is not None
        assert config.client._call_count == 2

    @pytest.mark.asyncio
    async def test_concurrent_fatal_from_two_observers(self) -> None:
        """Two observers both emit FATAL in the same turn — handled gracefully."""
        config = _RecordingConfig(
            ToolCallEvent(name="echo_tool", arguments='{"value": "x"}'),
            "should-not-reach",
        )

        class _FatalObs(BaseObserver):
            def __init__(self, name: str, msg: str) -> None:
                super().__init__(name, watch=EventWatch(ModelResponse))
                self._msg = msg

            async def process(self, events: list[BaseEvent], ctx: ContextType) -> ObserverAlert | None:
                return ObserverAlert(
                    source=self.name,
                    severity=Severity.FATAL,
                    message=self._msg,
                )

        agent = Agent(
            "test",
            config=config,
            observers=[_FatalObs("obs-a", "crash-a"), _FatalObs("obs-b", "crash-b")],
            tools=[echo_tool],
            assembly=[AlertPolicy()],
        )
        reply = await agent.ask("Hi")

        assert reply.body is not None
        assert "HALTED" in reply.body


class TestAlertPolicyOrdering:
    """Assembly policy ordering validation recognizes AlertPolicy as injection."""

    def test_alert_before_reduction_no_warning(self) -> None:
        """AlertPolicy before SlidingWindowPolicy produces no warnings."""
        from autogen.beta.assembly import AssemblerMiddleware
        from autogen.beta.policies import SlidingWindowPolicy

        policies = [AlertPolicy(), SlidingWindowPolicy(50)]
        warnings = AssemblerMiddleware.validate_order(policies)
        assert warnings == []

    def test_reduction_before_alert_warns(self) -> None:
        """SlidingWindowPolicy before AlertPolicy produces a warning."""
        from autogen.beta.assembly import AssemblerMiddleware
        from autogen.beta.policies import SlidingWindowPolicy

        policies = [SlidingWindowPolicy(50), AlertPolicy()]
        warnings = AssemblerMiddleware.validate_order(policies)
        assert len(warnings) == 1
        assert "alert" in warnings[0]
