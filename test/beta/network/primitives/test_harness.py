# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta.context import Context
from autogen.beta.events import ModelMessage, ModelRequest, ModelResponse
from autogen.beta.events.tool_events import ToolCallEvent, ToolResultEvent
from autogen.beta.network.events import DelegationResult, SchedulerTriggerFired
from autogen.beta.network.primitives.harness import (
    ConversationHarness,
    FormattedEvent,
    HarnessMiddleware,
    NetworkHarness,
)
from autogen.beta.network.primitives.signal import Severity, Signal
from autogen.beta.stream import MemoryStream


class TestConversationHarness:
    def test_selects_conversation_events(self) -> None:
        harness = ConversationHarness()
        events = [
            ModelRequest(content="hello"),
            ModelResponse(message=ModelMessage(content="hi")),
            ToolCallEvent(name="search", arguments="{}"),
            ToolResultEvent(id="1", name="search", content="result"),
            Signal(source="mon", severity=Severity.WARNING, message="warn"),
        ]
        selected = harness.select(events)
        # Should include ModelRequest, ModelResponse, ToolCallEvent, ToolResultEvent
        # Should NOT include Signal
        assert len(selected) == 4
        assert all(not isinstance(e, Signal) for e in selected)

    def test_format_returns_none(self) -> None:
        harness = ConversationHarness()
        event = ModelRequest(content="hello")
        assert harness.format(event) is None


class TestNetworkHarness:
    def test_includes_signals(self) -> None:
        harness = NetworkHarness()
        signal = Signal(source="mon", severity=Severity.CRITICAL, message="alert")
        events = [
            ModelRequest(content="hello"),
            signal,
        ]
        selected = harness.select(events)
        assert signal in selected

    def test_formats_signal(self) -> None:
        harness = NetworkHarness()
        signal = Signal(source="mon", severity=Severity.WARNING, message="high usage")
        formatted = harness.format(signal)
        assert formatted is not None
        assert "[SIGNAL/WARNING]" in formatted
        assert "mon" in formatted
        assert "high usage" in formatted

    def test_format_non_network_returns_none(self) -> None:
        harness = NetworkHarness()
        event = ModelRequest(content="hello")
        assert harness.format(event) is None

    def test_includes_formatted_events(self) -> None:
        harness = NetworkHarness()
        fe = FormattedEvent(content="[DELEGATION] done")
        events = [
            ModelRequest(content="hello"),
            fe,
        ]
        selected = harness.select(events)
        assert fe in selected

    def test_includes_delegation_result(self) -> None:
        harness = NetworkHarness()
        dr = DelegationResult(source="a", target="b", result="done")
        events = [
            ModelRequest(content="hello"),
            dr,
        ]
        selected = harness.select(events)
        assert dr in selected

    def test_formats_delegation_result(self) -> None:
        harness = NetworkHarness()
        dr = DelegationResult(source="researcher", target="writer", result="report written")
        formatted = harness.format(dr)
        assert formatted is not None
        assert "[DELEGATION RESULT]" in formatted
        assert "researcher" in formatted
        assert "writer" in formatted
        assert "report written" in formatted

    def test_includes_scheduler_trigger(self) -> None:
        harness = NetworkHarness()
        st = SchedulerTriggerFired(watch_id="w1", target="monitor", task="check")
        events = [
            ModelRequest(content="hello"),
            st,
        ]
        selected = harness.select(events)
        assert st in selected

    def test_formats_scheduler_trigger(self) -> None:
        harness = NetworkHarness()
        st = SchedulerTriggerFired(watch_id="w1", target="monitor", task="check")
        formatted = harness.format(st)
        assert formatted is not None
        assert "[SCHEDULED]" in formatted
        assert "w1" in formatted
        assert "monitor" in formatted

    def test_formatted_event_carries_original(self) -> None:
        original = ModelRequest(content="hello")
        fe = FormattedEvent(content="formatted", original=original)
        assert fe.original is original


class TestHarnessMiddleware:
    @pytest.mark.asyncio
    async def test_filters_events_for_llm(self) -> None:
        """HarnessMiddleware should filter events via harness.select before calling LLM."""
        stream = MemoryStream()
        ctx = Context(stream=stream)
        initial_event = ModelRequest(content="start")

        harness = ConversationHarness()
        mw = HarnessMiddleware(initial_event, ctx, harness=harness)

        # Simulate events that include non-conversation types
        all_events = [
            ModelRequest(content="hello"),
            ModelResponse(message=ModelMessage(content="hi")),
            Signal(source="mon", severity=Severity.WARNING, message="warn"),
        ]

        received_events = None

        async def mock_llm_call(events, context):
            nonlocal received_events
            received_events = list(events)
            return ModelResponse(message=ModelMessage(content="response"))

        await mw.on_llm_call(mock_llm_call, all_events, ctx)

        # Signal should be filtered out
        assert received_events is not None
        assert len(received_events) == 2
        assert all(not isinstance(e, Signal) for e in received_events)

    @pytest.mark.asyncio
    async def test_formats_events(self) -> None:
        """HarnessMiddleware should format events via harness.format."""
        stream = MemoryStream()
        ctx = Context(stream=stream)
        initial_event = ModelRequest(content="start")

        harness = NetworkHarness()
        mw = HarnessMiddleware(initial_event, ctx, harness=harness)

        signal = Signal(source="mon", severity=Severity.WARNING, message="alert")
        all_events = [
            ModelRequest(content="hello"),
            signal,
        ]

        received_events = None

        async def mock_llm_call(events, context):
            nonlocal received_events
            received_events = list(events)
            return ModelResponse(message=ModelMessage(content="response"))

        await mw.on_llm_call(mock_llm_call, all_events, ctx)

        # Signal should be formatted as FormattedEvent
        formatted = [e for e in received_events if isinstance(e, FormattedEvent)]
        assert len(formatted) == 1
        assert "[SIGNAL/WARNING]" in formatted[0].content
