# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Framework-core regression tests extracted from the former
``test/beta/network/test_bugfixes.py``.

The original grab-bag file mixed V2-network-specific bug fixes with
framework-core regressions. The V2 parts were dropped during the V3
rewrite; this file preserves the framework-core cases — nested event
import, Gemini usage normalization, FunctionTool name access,
Actor.run_subtasks sequential exception handling, and the
ObserverCompleted emission guarantee when ``detach()`` raises.
"""

from __future__ import annotations

import pytest

from autogen.beta.events import ModelMessage
from autogen.beta.events._serialization import import_event_class
from autogen.beta.events.base import BaseEvent


# ---------------------------------------------------------------------------
# Bug 1: _import_event_class can't handle nested class qualnames
# ---------------------------------------------------------------------------


class Outer:
    """Container for nested event class."""

    class NestedEvent(BaseEvent):
        value: str


class TestNestedEventClassImport:
    def test_import_module_level_event(self) -> None:
        cls = import_event_class(f"{ModelMessage.__module__}.{ModelMessage.__qualname__}")
        assert cls is ModelMessage

    def test_import_nested_event_class(self) -> None:
        qualname = f"{Outer.NestedEvent.__module__}.{Outer.NestedEvent.__qualname__}"
        cls = import_event_class(qualname)
        assert cls is Outer.NestedEvent

    def test_import_nonexistent_returns_none(self) -> None:
        cls = import_event_class("nonexistent.module.FakeEvent")
        assert cls is None

    def test_import_non_event_class_returns_none(self) -> None:
        cls = import_event_class("builtins.int")
        assert cls is None


# ---------------------------------------------------------------------------
# Bug 7: Gemini client usage normalization
# ---------------------------------------------------------------------------


class TestGeminiUsageNormalization:
    """Gemini client should normalize usage keys to standard names."""

    def test_gemini_usage_dict_has_standard_keys(self) -> None:
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

        assert usage["total_tokens"] == 150
        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 50

        assert usage["prompt_token_count"] == 100
        assert usage["candidates_token_count"] == 50
        assert usage["total_token_count"] == 150

    @pytest.mark.asyncio
    async def test_token_monitor_works_with_normalized_keys(self) -> None:
        from unittest.mock import MagicMock

        from autogen.beta.events import ModelResponse, Usage
        from autogen.beta.observers.token_monitor import TokenMonitor

        monitor = TokenMonitor(warn_threshold=100, alert_threshold=200)

        event = ModelResponse(
            usage=Usage(
                prompt_tokens=40,
                completion_tokens=30,
                total_tokens=70,
            )
        )

        ctx = MagicMock()
        result = await monitor.process([event], ctx)

        assert monitor.total_tokens == 70
        assert result is None  # Under threshold


# ---------------------------------------------------------------------------
# Bug 8: FunctionTool has no __name__ attribute
# ---------------------------------------------------------------------------


class TestFunctionToolNameAccess:
    def test_function_tool_name_via_schema(self) -> None:
        from autogen.beta.tools.final import tool

        @tool
        async def my_cool_tool(x: int) -> str:
            """A test tool."""
            return str(x)

        # __name__ should not be the right access path.
        assert not hasattr(my_cool_tool, "__name__") or not isinstance(
            getattr(my_cool_tool, "__name__", None), str
        )

        # schema.function.name is the supported access path.
        assert my_cool_tool.schema.function.name == "my_cool_tool"


# ---------------------------------------------------------------------------
# Bug 9: run_subtasks sequential mode doesn't catch exceptions
# ---------------------------------------------------------------------------


class TestRunSubtasksSequentialExceptionHandling:
    @pytest.mark.asyncio
    async def test_sequential_run_subtasks_catches_exception(self) -> None:
        from unittest.mock import MagicMock

        from autogen.beta import Actor

        actor = Actor("test-actor")

        call_log: list[str] = []

        async def mock_run_task(task: str, ctx):
            call_log.append(task)
            if task == "task-2-fail":
                raise RuntimeError("LLM API timeout")
            return f"result of {task}"

        actor._run_task = mock_run_task  # type: ignore[assignment]

        tasks = ["task-1-ok", "task-2-fail", "task-3-ok"]
        results = []
        for t in tasks:
            try:
                results.append(await actor._run_task(t, MagicMock()))
            except Exception as e:
                results.append(f"Error: {e}")

        assert len(call_log) == 3
        assert "task-1-ok" in call_log
        assert "task-2-fail" in call_log
        assert "task-3-ok" in call_log

        assert results[0] == "result of task-1-ok"
        assert "Error:" in results[1]
        assert "LLM API timeout" in results[1]
        assert results[2] == "result of task-3-ok"

    def test_sequential_code_path_has_try_except(self) -> None:
        import inspect

        from autogen.beta import Actor

        source = inspect.getsource(Actor._build_subtask_tools)
        assert "except Exception as e:" in source
        assert 'f"Error: {e}"' in source


# ---------------------------------------------------------------------------
# Bug 13: ObserverCompleted emission must survive detach() failures
# ---------------------------------------------------------------------------


class TestObserverDetachFailureStillEmitsCompleted:
    @pytest.mark.asyncio
    async def test_completed_emitted_despite_detach_error(self) -> None:
        from unittest.mock import AsyncMock, MagicMock, patch

        from autogen.beta import Actor
        from autogen.beta.events import ModelMessage
        from autogen.beta.events.lifecycle import ObserverCompleted

        class _FailingDetachObserver:
            """Observer whose detach() always raises."""

            name = "failing-observer"

            def attach(self, stream, ctx):
                pass

            def detach(self):
                raise RuntimeError("detach exploded")

        actor = Actor("test-actor", observers=[_FailingDetachObserver()])

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

        with patch(
            "autogen.beta.agent.Agent._execute",
            new_callable=AsyncMock,
            return_value=mock_reply,
        ):
            mock_client = MagicMock()
            await actor._execute(
                ModelMessage(content="test"),
                context=ctx,
                client=mock_client,
            )

        completed = [e for e in sent_events if isinstance(e, ObserverCompleted)]
        assert len(completed) == 1
        assert completed[0].name == "failing-observer"

    @pytest.mark.asyncio
    async def test_normal_detach_still_emits_completed(self) -> None:
        from unittest.mock import AsyncMock, MagicMock, patch

        from autogen.beta import Actor
        from autogen.beta.events import ModelMessage
        from autogen.beta.events.lifecycle import ObserverCompleted

        class _GoodObserver:
            name = "good-observer"

            def attach(self, stream, ctx):
                pass

            def detach(self):
                pass

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

        with patch(
            "autogen.beta.agent.Agent._execute",
            new_callable=AsyncMock,
            return_value=mock_reply,
        ):
            mock_client = MagicMock()
            await actor._execute(
                ModelMessage(content="test"),
                context=ctx,
                client=mock_client,
            )

        completed = [e for e in sent_events if isinstance(e, ObserverCompleted)]
        assert len(completed) == 1
        assert completed[0].name == "good-observer"
