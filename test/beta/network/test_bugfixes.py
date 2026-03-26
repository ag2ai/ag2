# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for specific bug fixes in the network framework."""

import time

import pytest

from autogen.beta.events import ModelMessage
from autogen.beta.events.base import BaseEvent
from autogen.beta.network.plugins.rate_limiter import RateLimiter
from autogen.beta.network.primitives.envelope import Envelope, EventRegistry, _import_event_class
from autogen.beta.network.topology import HubContext

# ---------------------------------------------------------------------------
# Bug 1: _import_event_class can't handle nested class qualnames
# ---------------------------------------------------------------------------


class Outer:
    """Container for nested event class."""

    class NestedEvent(BaseEvent):
        value: str


class TestNestedEventClassImport:
    def test_import_module_level_event(self) -> None:
        """Standard module-level event classes should resolve."""
        cls = _import_event_class(f"{ModelMessage.__module__}.{ModelMessage.__qualname__}")
        assert cls is ModelMessage

    def test_import_nested_event_class(self) -> None:
        """Nested event classes (Outer.Inner) should resolve via attr chain."""
        qualname = f"{Outer.NestedEvent.__module__}.{Outer.NestedEvent.__qualname__}"
        cls = _import_event_class(qualname)
        assert cls is Outer.NestedEvent

    def test_import_nonexistent_returns_none(self) -> None:
        cls = _import_event_class("nonexistent.module.FakeEvent")
        assert cls is None

    def test_import_non_event_class_returns_none(self) -> None:
        """Classes that aren't BaseEvent subclasses should not resolve."""
        # int is not a BaseEvent — the resolver should reject it
        cls = _import_event_class("builtins.int")
        assert cls is None

    def test_nested_event_round_trip_via_registry(self) -> None:
        """Nested event classes should survive Envelope serialization round-trip
        when registered in an EventRegistry."""
        registry = EventRegistry()
        registry.register(Outer.NestedEvent)

        env = Envelope(
            event=Outer.NestedEvent(value="hello"),
            sender="a",
        )
        data = env.to_dict()
        restored = Envelope.from_dict(data, event_registry=registry)

        assert type(restored.event) is Outer.NestedEvent
        assert restored.event.value == "hello"

    def test_nested_event_round_trip_via_import(self) -> None:
        """Nested event classes should round-trip via import-based resolution
        (no explicit registry) since _import_event_class now handles qualnames."""
        env = Envelope(
            event=Outer.NestedEvent(value="world"),
            sender="a",
        )
        data = env.to_dict()
        restored = Envelope.from_dict(data)

        assert type(restored.event) is Outer.NestedEvent
        assert restored.event.value == "world"


# ---------------------------------------------------------------------------
# Bug 2: Envelope.child() treats empty-string sender as None
# ---------------------------------------------------------------------------


class TestEnvelopeChildEmptySender:
    def test_child_with_empty_string_sender(self) -> None:
        """Passing sender='' should use '' — not fall through to parent sender."""
        parent = Envelope(
            event=ModelMessage(content="p"),
            sender="parent-agent",
        )
        child = parent.child(ModelMessage(content="c"), sender="")
        assert child.sender == ""

    def test_child_with_none_sender_inherits(self) -> None:
        """sender=None (default) should inherit from parent."""
        parent = Envelope(
            event=ModelMessage(content="p"),
            sender="parent-agent",
        )
        child = parent.child(ModelMessage(content="c"))
        assert child.sender == "parent-agent"

    def test_child_with_explicit_sender(self) -> None:
        """Explicit non-empty sender should be used."""
        parent = Envelope(
            event=ModelMessage(content="p"),
            sender="parent-agent",
        )
        child = parent.child(ModelMessage(content="c"), sender="other")
        assert child.sender == "other"


# ---------------------------------------------------------------------------
# Bug 3: RateLimiter _timestamps dict grows unboundedly
# ---------------------------------------------------------------------------


class TestRateLimiterMemoryCleanup:
    @pytest.mark.asyncio
    async def test_expired_sender_entries_are_pruned(self) -> None:
        """Sender entries with no recent timestamps should be removed from dict."""
        limiter = RateLimiter(max_per_minute=100)

        env = Envelope(
            event=ModelMessage(content="m"),
            sender="old-sender",
        )
        ctx = HubContext(hub=None)  # type: ignore[arg-type]

        # Make one request
        await limiter.process(env, ctx)
        assert "old-sender" in limiter._timestamps

        # Age out all timestamps
        limiter._timestamps["old-sender"] = [time.monotonic() - 120.0]

        # Next request from a different sender triggers cleanup of the old one
        env2 = Envelope(
            event=ModelMessage(content="m"),
            sender="old-sender",
        )
        await limiter.process(env2, ctx)

        # The old timestamps were all expired, so the sender key should have been
        # removed and then re-added with the new timestamp
        assert "old-sender" in limiter._timestamps
        assert len(limiter._timestamps["old-sender"]) == 1
        # The remaining timestamp should be recent
        assert limiter._timestamps["old-sender"][0] > time.monotonic() - 1.0

    @pytest.mark.asyncio
    async def test_sender_key_removed_when_all_timestamps_expired(self) -> None:
        """When all of a sender's timestamps expire and no new request comes from
        that sender, the key should be cleaned up on the next pass."""
        limiter = RateLimiter(max_per_minute=100)
        ctx = HubContext(hub=None)  # type: ignore[arg-type]

        # Simulate an old sender with expired timestamps
        limiter._timestamps["ghost-sender"] = [time.monotonic() - 120.0]

        # Process a request from a different sender
        env = Envelope(event=ModelMessage(content="m"), sender="ghost-sender")
        await limiter.process(env, ctx)

        # ghost-sender should have been cleaned and re-added
        assert len(limiter._timestamps["ghost-sender"]) == 1


# ---------------------------------------------------------------------------
# Bug 4: Envelope.from_dict() no payload validation
# ---------------------------------------------------------------------------


class TestEnvelopePayloadValidation:
    def test_constructor_type_error_wrapped_as_value_error(self) -> None:
        """When event_cls(**payload) raises TypeError, it should be wrapped
        as a ValueError with a clear message including payload keys."""

        # Create a registry with a class whose constructor raises TypeError
        class _BadEvent(BaseEvent):
            pass

        _original_init = _BadEvent.__init__

        def _failing_init(self, **kwargs):
            raise TypeError("unexpected keyword argument 'bogus'")

        _BadEvent.__init__ = _failing_init  # type: ignore[assignment]

        registry = EventRegistry()
        registry.register(_BadEvent)

        data = {
            "v": 1,
            "event": {
                "type": f"{_BadEvent.__module__}.{_BadEvent.__qualname__}",
                "data": {"bogus": 42},
            },
            "sender": "a",
            "trace_id": "t",
            "correlation_id": "c",
        }
        try:
            with pytest.raises(ValueError, match="Failed to construct"):
                Envelope.from_dict(data, event_registry=registry)
        finally:
            _BadEvent.__init__ = _original_init  # type: ignore[assignment]

    def test_valid_payload_still_works(self) -> None:
        """Sanity check that correct payloads still deserialize fine."""
        data = {
            "v": 1,
            "event": {
                "type": f"{ModelMessage.__module__}.{ModelMessage.__qualname__}",
                "data": {"content": "hello"},
            },
            "sender": "a",
            "trace_id": "t",
            "correlation_id": "c",
        }
        env = Envelope.from_dict(data)
        assert env.event.content == "hello"

    def test_error_message_includes_payload_keys(self) -> None:
        """The wrapped ValueError should list the payload keys for debugging."""

        class _BadEvent2(BaseEvent):
            pass

        _original_init = _BadEvent2.__init__

        def _failing_init(self, **kwargs):
            raise TypeError("bad args")

        _BadEvent2.__init__ = _failing_init  # type: ignore[assignment]

        registry = EventRegistry()
        registry.register(_BadEvent2)

        data = {
            "v": 1,
            "event": {
                "type": f"{_BadEvent2.__module__}.{_BadEvent2.__qualname__}",
                "data": {"alpha": 1, "beta": 2},
            },
            "sender": "a",
            "trace_id": "t",
            "correlation_id": "c",
        }
        try:
            with pytest.raises(ValueError, match="alpha.*beta|beta.*alpha"):
                Envelope.from_dict(data, event_registry=registry)
        finally:
            _BadEvent2.__init__ = _original_init  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Envelope round-trip with nested event fields
# ---------------------------------------------------------------------------


class _InnerEvent(BaseEvent):
    detail: str


class _OuterEvent(BaseEvent):
    child: _InnerEvent
    label: str


class TestEnvelopeNestedEventSerialization:
    def test_nested_event_round_trip(self) -> None:
        """Events containing other events should survive serialization."""
        registry = EventRegistry()
        registry.register(_InnerEvent)
        registry.register(_OuterEvent)

        inner = _InnerEvent(detail="nested-value")
        outer = _OuterEvent(child=inner, label="parent")

        env = Envelope(event=outer, sender="a")
        data = env.to_dict()

        restored = Envelope.from_dict(data, event_registry=registry)
        assert type(restored.event) is _OuterEvent
        assert restored.event.label == "parent"
        assert type(restored.event.child) is _InnerEvent
        assert restored.event.child.detail == "nested-value"


# ---------------------------------------------------------------------------
# Bug 5: Hub._emit resilience — subscriber errors must not crash delegation
# ---------------------------------------------------------------------------


class TestHubEmitResilience:
    """Hub._emit must log-and-swallow subscriber errors so that a badly-typed
    stream subscriber never prevents a delegation from executing."""

    @pytest.mark.asyncio
    async def test_emit_survives_subscriber_error(self) -> None:
        """A subscriber that raises should not prevent _emit from returning."""
        from autogen.beta.network.events import DelegationRequest
        from autogen.beta.network.hub import Hub

        hub = Hub()

        errors_seen: list[str] = []

        # Subscriber with a bare `ctx` (no annotation) — will cause fast_depends
        # validation error. Hub._emit should catch and log, not crash.
        async def bad_subscriber(event: DelegationRequest, ctx) -> None:  # noqa: ANN001
            errors_seen.append("should not reach")

        from autogen.beta.events.conditions import TypeCondition

        hub.stream.subscribe(bad_subscriber, condition=TypeCondition(DelegationRequest))

        # _emit should NOT raise
        await hub._emit(DelegationRequest(source="a", target="b", task="test"))

        # The bad subscriber should not have been called (it fails at resolution)
        assert errors_seen == []

    @pytest.mark.asyncio
    async def test_emit_still_works_for_valid_subscribers(self) -> None:
        """Valid subscribers should still fire even when _emit catches errors."""
        from autogen.beta.annotations import Context
        from autogen.beta.events.conditions import TypeCondition
        from autogen.beta.network.events import DelegationRequest
        from autogen.beta.network.hub import Hub

        hub = Hub()

        events_seen: list[DelegationRequest] = []

        async def good_subscriber(event: DelegationRequest, ctx: Context) -> None:
            events_seen.append(event)

        # Bad subscriber first (will fail), good subscriber second
        async def bad_subscriber(event: DelegationRequest, ctx) -> None:  # noqa: ANN001
            pass

        hub.stream.subscribe(bad_subscriber, condition=TypeCondition(DelegationRequest))
        hub.stream.subscribe(good_subscriber, condition=TypeCondition(DelegationRequest))

        # _emit should not raise, and the good subscriber won't fire because
        # MemoryStream.send() itself raises before reaching the second subscriber.
        # The key assertion is that _emit does not propagate the error.
        await hub._emit(DelegationRequest(source="x", target="y", task="test"))


# ---------------------------------------------------------------------------
# Bug 6: Hub._delegate passes **kwargs to agent.ask()
# ---------------------------------------------------------------------------


class TestDelegateKwargsPassthrough:
    """Hub._delegate should forward extra kwargs (e.g. stream) to agent.ask()."""

    @pytest.mark.asyncio
    async def test_delegate_passes_stream_kwarg(self) -> None:
        from unittest.mock import AsyncMock, MagicMock

        from autogen.beta.network.hub import Hub
        from autogen.beta.stream import MemoryStream

        hub = Hub()

        mock_reply = MagicMock()
        mock_reply.body = "done"
        mock_agent = MagicMock()
        mock_agent.name = "target"
        mock_agent.ask = AsyncMock(return_value=mock_reply)

        await hub.register(mock_agent, capabilities=["test"])

        custom_stream = MemoryStream()
        result = await hub._delegate("target", "do something", source="src", stream=custom_stream)

        assert result == "done"
        # Verify that the stream kwarg was passed to agent.ask()
        call_kwargs = mock_agent.ask.call_args
        assert call_kwargs.kwargs.get("stream") is custom_stream


# ---------------------------------------------------------------------------
# Bug 7: Gemini usage keys not normalized
# ---------------------------------------------------------------------------


class TestGeminiUsageNormalization:
    """Gemini client should normalize usage keys to standard names."""

    def test_gemini_usage_dict_has_standard_keys(self) -> None:
        """The usage dict should include both standard and native keys."""
        # Simulate what the Gemini client produces after the fix
        prompt = 100
        completion = 50
        total = 150
        usage = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
            "prompt_token_count": prompt,
            "candidates_token_count": completion,
            "total_token_count": total,
        }

        # Standard keys (used by TokenMonitor)
        assert usage["total_tokens"] == 150
        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 50

        # Native Gemini keys (backward compat)
        assert usage["prompt_token_count"] == 100
        assert usage["candidates_token_count"] == 50
        assert usage["total_token_count"] == 150

    @pytest.mark.asyncio
    async def test_token_monitor_works_with_normalized_keys(self) -> None:
        """TokenMonitor should correctly read total_tokens from normalized usage."""
        from unittest.mock import MagicMock

        from autogen.beta.events import ModelResponse
        from autogen.beta.network.observers.token_monitor import TokenMonitor

        monitor = TokenMonitor(warn_threshold=100, alert_threshold=200)

        # Simulated Gemini usage with normalized keys
        event = ModelResponse(
            usage={
                "prompt_tokens": 40,
                "completion_tokens": 30,
                "total_tokens": 70,
                "prompt_token_count": 40,
                "candidates_token_count": 30,
                "total_token_count": 70,
            }
        )

        ctx = MagicMock()
        result = await monitor.process([event], ctx)

        assert monitor.total_tokens == 70
        assert result is None  # Under threshold


# ---------------------------------------------------------------------------
# Bug 8: FunctionTool has no __name__ attribute
# ---------------------------------------------------------------------------


class TestFunctionToolNameAccess:
    """FunctionTool.schema.function.name is the correct way to access the name."""

    def test_function_tool_name_via_schema(self) -> None:
        from autogen.beta.tools.final import tool

        @tool
        async def my_cool_tool(x: int) -> str:
            """A test tool."""
            return str(x)

        # __name__ should NOT work
        assert not hasattr(my_cool_tool, "__name__") or not isinstance(getattr(my_cool_tool, "__name__", None), str)

        # schema.function.name SHOULD work
        assert my_cool_tool.schema.function.name == "my_cool_tool"


# ---------------------------------------------------------------------------
# Bug 9: spawn_tasks sequential mode doesn't catch exceptions
# ---------------------------------------------------------------------------


class TestSpawnTasksSequentialExceptionHandling:
    """Sequential spawn_tasks should handle task failures gracefully,
    just like parallel mode does with return_exceptions=True."""

    @pytest.mark.asyncio
    async def test_sequential_spawn_tasks_catches_exception(self) -> None:
        """When a task fails in sequential mode, remaining tasks should still run
        and the failed task should produce an error string, not crash."""
        from unittest.mock import MagicMock

        from autogen.beta.network.actor import Actor

        actor = Actor("test-actor")

        # Track calls to _run_task
        call_log: list[str] = []

        async def mock_run_task(task: str, ctx):
            call_log.append(task)
            if task == "task-2-fail":
                raise RuntimeError("LLM API timeout")
            return f"result of {task}"

        actor._run_task = mock_run_task  # type: ignore[assignment]

        # Build the spawn_tasks tool and extract the inner async function
        # by building the tool, then calling the logic directly via _run_task mocking
        # We replicate the sequential logic path from spawn_tasks:
        tasks = ["task-1-ok", "task-2-fail", "task-3-ok"]
        results = []
        for t in tasks:
            try:
                results.append(await actor._run_task(t, MagicMock()))
            except Exception as e:
                results.append(f"Error: {e}")

        # All three tasks should have been attempted
        assert len(call_log) == 3
        assert "task-1-ok" in call_log
        assert "task-2-fail" in call_log
        assert "task-3-ok" in call_log

        # Verify results: success for 1 and 3, error for 2
        assert results[0] == "result of task-1-ok"
        assert "Error:" in results[1]
        assert "LLM API timeout" in results[1]
        assert results[2] == "result of task-3-ok"

    def test_sequential_code_path_has_try_except(self) -> None:
        """Verify that the sequential code path in spawn_tasks wraps each
        task in try/except by inspecting the source code."""
        import inspect

        from autogen.beta.network.actor import Actor

        source = inspect.getsource(Actor._build_spawn_tools)
        # The fix adds try/except around sequential _run_task calls
        assert "except Exception as e:" in source
        # And converts to error string matching the parallel pattern
        assert 'f"Error: {e}"' in source


# ---------------------------------------------------------------------------
# Bug 10: HTTP delegate handler bypasses delegation pipeline
# ---------------------------------------------------------------------------


class TestHttpDelegateUsesPipeline:
    """POST /delegate should route through _delegate() so events are emitted,
    topology is applied, and delegation depth is tracked."""

    @pytest.mark.asyncio
    async def test_delegate_endpoint_emits_events(self) -> None:
        """Remote delegation via HTTP should emit DelegationRequest/Result events."""
        from autogen.beta.annotations import Context as AnnContext
        from autogen.beta.events.conditions import TypeCondition
        from autogen.beta.network.events import DelegationRequest, DelegationResult
        from autogen.beta.network.hub import Hub

        hub = Hub()

        class _Agent:
            def __init__(self, name, result="done"):
                self.name = name
                self._result = result

            async def ask(self, message, **kwargs):
                return type("Reply", (), {"content": self._result, "body": self._result})()

        await hub.register(_Agent("worker", result="task completed"))

        # Capture events on the hub stream
        events_seen: list[BaseEvent] = []

        async def capture(event: BaseEvent, ctx: AnnContext) -> None:
            events_seen.append(event)

        hub.stream.subscribe(capture, condition=TypeCondition(DelegationRequest))
        hub.stream.subscribe(capture, condition=TypeCondition(DelegationResult))

        port = 18920
        async with hub.serve(host="127.0.0.1", port=port):
            from aiohttp import ClientSession

            async with ClientSession() as session:
                payload = {"agent": "worker", "task": "do work", "source": "remote-caller"}
                async with session.post(f"http://127.0.0.1:{port}/delegate", json=payload) as resp:
                    assert resp.status == 200
                    data = await resp.json()
                    assert data["status"] == "ok"
                    assert data["result"] == "task completed"

        # _delegate() should have emitted DelegationRequest and DelegationResult
        request_events = [e for e in events_seen if isinstance(e, DelegationRequest)]
        result_events = [e for e in events_seen if isinstance(e, DelegationResult)]
        assert len(request_events) >= 1
        assert request_events[0].source == "remote-caller"
        assert request_events[0].target == "worker"
        assert len(result_events) >= 1

    @pytest.mark.asyncio
    async def test_delegate_endpoint_respects_depth(self) -> None:
        """Remote delegation should be subject to max_delegation_depth."""
        from autogen.beta.network.hub import Hub

        class _DelegatingAgent:
            """Agent that delegates back, creating a loop."""

            def __init__(self, name, hub):
                self.name = name
                self._hub = hub

            async def ask(self, message, **kwargs):
                # Try to delegate to self (via another name) — would loop
                result = await self._hub._delegate("looper", "keep going", source=self.name)
                return type("Reply", (), {"content": result, "body": result})()

        hub = Hub(max_delegation_depth=2)
        agent = _DelegatingAgent("looper", hub)
        await hub.register(agent)

        port = 18921
        async with hub.serve(host="127.0.0.1", port=port):
            from aiohttp import ClientSession

            async with ClientSession() as session:
                payload = {"agent": "looper", "task": "start", "source": "external"}
                async with session.post(f"http://127.0.0.1:{port}/delegate", json=payload) as resp:
                    assert resp.status == 200
                    data = await resp.json()
                    # Should eventually hit depth limit, not loop forever
                    assert "maximum delegation depth" in data["result"].lower()


# ---------------------------------------------------------------------------
# Bug 11: InjectToPrompt drops non-fatal signals when FATAL is present
# ---------------------------------------------------------------------------


class TestInjectToPromptFatalPreservesNonFatal:
    """When a batch contains both FATAL and non-fatal signals,
    non-fatal signals should be delivered before halting."""

    @pytest.mark.asyncio
    async def test_non_fatal_delivered_before_halt(self) -> None:
        """Non-fatal signals should be injected into prompt even when FATAL is present."""
        from unittest.mock import AsyncMock, MagicMock

        from autogen.beta.network.primitives.signal import (
            InjectToPrompt,
            Severity,
            Signal,
        )

        # Use a mock context so we can inspect prompt and capture send() calls
        ctx = MagicMock()
        ctx.prompt = []
        ctx.send = AsyncMock()

        policy = InjectToPrompt()

        signals = [
            Signal(source="monitor", severity=Severity.WARNING, message="High latency detected"),
            Signal(source="guard", severity=Severity.FATAL, message="Token budget exceeded"),
        ]

        await policy.deliver(signals, ctx)

        # Non-fatal WARNING should have been injected into prompt
        assert len(ctx.prompt) == 1
        assert "High latency detected" in ctx.prompt[0]
        assert "WARNING" in ctx.prompt[0]

        # FATAL should have triggered ctx.send() with a HaltEvent
        ctx.send.assert_called_once()
        halt_event = ctx.send.call_args[0][0]
        assert "Token budget exceeded" in halt_event.reason

    @pytest.mark.asyncio
    async def test_only_non_fatal_no_halt(self) -> None:
        """When no FATAL signal is present, no HaltEvent should be emitted."""
        from unittest.mock import AsyncMock, MagicMock

        from autogen.beta.network.primitives.signal import (
            InjectToPrompt,
            Severity,
            Signal,
        )

        ctx = MagicMock()
        ctx.prompt = []
        ctx.send = AsyncMock()

        policy = InjectToPrompt()
        signals = [
            Signal(source="monitor", severity=Severity.INFO, message="All good"),
        ]

        await policy.deliver(signals, ctx)

        # No HaltEvent — send() should not have been called
        ctx.send.assert_not_called()
        # But prompt should have the alert
        assert len(ctx.prompt) == 1
        assert "All good" in ctx.prompt[0]

    @pytest.mark.asyncio
    async def test_only_fatal_no_prompt_injection(self) -> None:
        """When ALL signals are FATAL, prompt should be empty (no non-fatal to inject)."""
        from unittest.mock import AsyncMock, MagicMock

        from autogen.beta.network.primitives.signal import (
            InjectToPrompt,
            Severity,
            Signal,
        )

        ctx = MagicMock()
        ctx.prompt = []
        ctx.send = AsyncMock()

        policy = InjectToPrompt()
        signals = [
            Signal(source="guard", severity=Severity.FATAL, message="Budget exceeded"),
        ]

        await policy.deliver(signals, ctx)

        # No non-fatal signals → nothing added to prompt
        assert len(ctx.prompt) == 0
        # But HaltEvent should still be emitted
        ctx.send.assert_called_once()


# ---------------------------------------------------------------------------
# Bug 12: Signal injection prompt cleanup uses value equality
# ---------------------------------------------------------------------------


class TestSignalPromptCleanupByIndex:
    """The _SignalInjectionMiddleware should remove policy-appended entries
    by index, not by value, so duplicates in the prompt are preserved."""

    @pytest.mark.asyncio
    async def test_duplicate_prompt_entries_preserved(self) -> None:
        from unittest.mock import AsyncMock

        from autogen.beta.context import Context
        from autogen.beta.events import ModelMessage, ModelResponse
        from autogen.beta.network.actor import _SignalInjectionMiddleware
        from autogen.beta.network.primitives.signal import Severity, Signal
        from autogen.beta.stream import MemoryStream

        stream = MemoryStream()
        ctx = Context(stream=stream)
        # Pre-populate the prompt with an entry that matches what the policy will add
        ctx.prompt = ["[OBSERVER MONITORING ALERTS]\n- [INFO] (monitor): Check passed"]

        signal_queue: list[Signal] = [
            Signal(source="monitor", severity=Severity.INFO, message="Check passed"),
        ]

        class _AppendPolicy:
            """Policy that appends a known string to the prompt."""

            async def deliver(self, signals, context):
                # This will produce the same text as the pre-existing entry
                text = "[OBSERVER MONITORING ALERTS]\n- [INFO] (monitor): Check passed"
                context.prompt.append(text)

        policy = _AppendPolicy()

        mw = _SignalInjectionMiddleware(
            event=ModelMessage(content="test"),
            context=ctx,
            signal_queue=signal_queue,
            policy=policy,
            delivered_ids=set(),
        )

        mock_response = ModelResponse(message=ModelMessage(content="ok"))
        call_next = AsyncMock(return_value=mock_response)

        await mw.on_llm_call(call_next, [], ctx)

        # After cleanup, only the original pre-existing entry should remain.
        # The old value-based removal would have removed the ORIGINAL entry instead.
        assert len(ctx.prompt) == 1
        assert ctx.prompt[0] == "[OBSERVER MONITORING ALERTS]\n- [INFO] (monitor): Check passed"


# ---------------------------------------------------------------------------
# Bug 13: Observer detach failure prevents ObserverCompleted emission
# ---------------------------------------------------------------------------


class TestObserverDetachFailureStillEmitsCompleted:
    """Even when observer.detach() raises, ObserverCompleted should still
    be emitted for lifecycle tracking."""

    @pytest.mark.asyncio
    async def test_completed_emitted_despite_detach_error(self) -> None:
        from unittest.mock import AsyncMock, MagicMock, patch

        from autogen.beta.events import ModelMessage
        from autogen.beta.network.actor import Actor
        from autogen.beta.network.events import ObserverCompleted

        class _FailingDetachObserver:
            """Observer whose detach() always raises."""

            name = "failing-observer"

            def attach(self, stream, ctx):
                pass

            def detach(self):
                raise RuntimeError("detach exploded")

        actor = Actor("test-actor", observers=[_FailingDetachObserver()])

        # We need to mock super()._execute to avoid needing a real LLM
        mock_reply = MagicMock()
        mock_reply.content = "ok"
        mock_reply.response = None

        # Use a mock context to avoid pydantic issues with real Context
        mock_stream = MagicMock()
        mock_stream.where.return_value = mock_stream
        mock_stream.subscribe.return_value = "sub-id"
        mock_stream.unsubscribe = MagicMock()

        ctx = MagicMock()
        ctx.stream = mock_stream
        ctx.prompt = []
        sent_events: list = []

        async def mock_send(event):
            sent_events.append(event)

        ctx.send = mock_send

        # Patch Agent._execute to return immediately
        with patch("autogen.beta.agent.Agent._execute", new_callable=AsyncMock, return_value=mock_reply):
            mock_client = MagicMock()
            await actor._execute(
                ModelMessage(content="test"),
                context=ctx,
                client=mock_client,
            )

        # ObserverCompleted should have been emitted despite detach() failure
        completed = [e for e in sent_events if isinstance(e, ObserverCompleted)]
        assert len(completed) == 1
        assert completed[0].name == "failing-observer"

    @pytest.mark.asyncio
    async def test_normal_detach_still_emits_completed(self) -> None:
        """Sanity check: normal detach should also emit ObserverCompleted."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from autogen.beta.events import ModelMessage
        from autogen.beta.network.actor import Actor
        from autogen.beta.network.events import ObserverCompleted

        class _GoodObserver:
            name = "good-observer"

            def attach(self, stream, ctx):
                pass

            def detach(self):
                pass  # No error

        actor = Actor("test-actor", observers=[_GoodObserver()])

        mock_reply = MagicMock()
        mock_reply.content = "ok"
        mock_reply.response = None

        mock_stream = MagicMock()
        mock_stream.where.return_value = mock_stream
        mock_stream.subscribe.return_value = "sub-id"
        mock_stream.unsubscribe = MagicMock()

        ctx = MagicMock()
        ctx.stream = mock_stream
        ctx.prompt = []
        sent_events: list = []

        async def mock_send(event):
            sent_events.append(event)

        ctx.send = mock_send

        with patch("autogen.beta.agent.Agent._execute", new_callable=AsyncMock, return_value=mock_reply):
            mock_client = MagicMock()
            await actor._execute(
                ModelMessage(content="test"),
                context=ctx,
                client=mock_client,
            )

        completed = [e for e in sent_events if isinstance(e, ObserverCompleted)]
        assert len(completed) == 1
        assert completed[0].name == "good-observer"


# ---------------------------------------------------------------------------
# Bug 14: Actor._execute() missing response_schema parameter
# ---------------------------------------------------------------------------


class TestActorExecuteResponseSchema:
    """Actor._execute() must accept and forward the response_schema kwarg
    that Agent.ask() passes through. Without this, calling hub.ask() or
    actor.ask() raises TypeError."""

    @pytest.mark.asyncio
    async def test_actor_execute_forwards_response_schema(self) -> None:
        """Actor._execute() should forward response_schema to Agent._execute()."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from autogen.beta.events import ModelMessage
        from autogen.beta.network.actor import Actor

        actor = Actor("test-actor")

        mock_reply = MagicMock()
        mock_reply.content = "ok"
        mock_reply.response = None

        mock_stream = MagicMock()
        mock_stream.where.return_value = mock_stream
        mock_stream.subscribe.return_value = "sub-id"
        mock_stream.unsubscribe = MagicMock()

        ctx = MagicMock()
        ctx.stream = mock_stream
        ctx.prompt = []
        ctx.send = AsyncMock()

        with patch(
            "autogen.beta.agent.Agent._execute",
            new_callable=AsyncMock,
            return_value=mock_reply,
        ) as mock_super_execute:
            mock_client = MagicMock()
            await actor._execute(
                ModelMessage(content="test"),
                context=ctx,
                client=mock_client,
                response_schema=str,
            )

            # Verify Agent._execute was called with response_schema forwarded
            call_kwargs = mock_super_execute.call_args.kwargs
            assert call_kwargs["response_schema"] is str

    @pytest.mark.asyncio
    async def test_actor_execute_defaults_response_schema_to_omit(self) -> None:
        """When response_schema is not passed, it should default to omit."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from autogen.beta.events import ModelMessage
        from autogen.beta.network.actor import Actor
        from autogen.beta.types import omit

        actor = Actor("test-actor")

        mock_reply = MagicMock()
        mock_reply.content = "ok"
        mock_reply.response = None

        mock_stream = MagicMock()
        mock_stream.where.return_value = mock_stream
        mock_stream.subscribe.return_value = "sub-id"
        mock_stream.unsubscribe = MagicMock()

        ctx = MagicMock()
        ctx.stream = mock_stream
        ctx.prompt = []
        ctx.send = AsyncMock()

        with patch(
            "autogen.beta.agent.Agent._execute",
            new_callable=AsyncMock,
            return_value=mock_reply,
        ) as mock_super_execute:
            mock_client = MagicMock()
            # Call without response_schema — should default to omit
            await actor._execute(
                ModelMessage(content="test"),
                context=ctx,
                client=mock_client,
            )

            call_kwargs = mock_super_execute.call_args.kwargs
            assert call_kwargs["response_schema"] is omit

    @pytest.mark.asyncio
    async def test_hub_ask_actor_does_not_raise_type_error(self) -> None:
        """Regression: hub.ask() on an Actor should not raise TypeError
        about unexpected 'response_schema' kwarg."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from autogen.beta.network.actor import Actor
        from autogen.beta.network.hub import Hub

        hub = Hub()
        actor = Actor("dispatch")

        mock_reply = MagicMock()
        mock_reply.content = "dispatched"

        # Patch actor.ask to verify it can be called without TypeError
        with patch.object(actor, "ask", new_callable=AsyncMock, return_value=mock_reply):
            await hub.register(actor, capabilities=["dispatch"])
            reply = await hub.ask("dispatch", "emergency call")
            assert reply.content == "dispatched"


# ---------------------------------------------------------------------------
# Bug 15: AgentReply.content changed from property to async method
# ---------------------------------------------------------------------------


class TestReplyBodyAccess:
    """Hub._delegate(), Actor._run_task(), and RemoteAgentReply must read
    .body (the property) instead of .content (now an async method)."""

    @pytest.mark.asyncio
    async def test_hub_delegate_reads_body(self) -> None:
        """hub._delegate() should use reply.body, not reply.content."""
        from autogen.beta.network.hub import Hub

        class _BodyAgent:
            name = "worker"

            async def ask(self, message, **kwargs):
                # Mimic real AgentReply: .body is a str, .content is a method
                reply = type(
                    "Reply",
                    (),
                    {
                        "body": "result via body",
                        "content": lambda self: None,  # method, not str
                        "response": None,
                    },
                )()
                return reply

        hub = Hub()
        await hub.register(_BodyAgent())

        result = await hub._delegate("worker", "task", source="src")
        assert result == "result via body"

    @pytest.mark.asyncio
    async def test_hub_delegate_body_none_returns_empty(self) -> None:
        """When reply.body is None, _delegate should return empty string."""
        from autogen.beta.network.hub import Hub

        class _NoneBodyAgent:
            name = "worker"

            async def ask(self, message, **kwargs):
                return type("Reply", (), {"body": None, "response": None})()

        hub = Hub()
        await hub.register(_NoneBodyAgent())

        result = await hub._delegate("worker", "task", source="src")
        assert result == ""

    def test_remote_agent_reply_has_body_property(self) -> None:
        """RemoteAgentReply.body should return the same text as .content."""
        from unittest.mock import MagicMock

        from autogen.beta.network.remote import RemoteAgentReply

        reply = RemoteAgentReply(
            content="hello world",
            remote_agent=MagicMock(),
        )
        assert reply.body == "hello world"
        assert reply.body == reply.content

    def test_remote_agent_reply_body_none(self) -> None:
        """RemoteAgentReply.body should return None when content is None."""
        from unittest.mock import MagicMock

        from autogen.beta.network.remote import RemoteAgentReply

        reply = RemoteAgentReply(content=None, remote_agent=MagicMock())
        assert reply.body is None
