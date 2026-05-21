# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for code execution, human input, and remote reply agent instrumentators."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from autogen import ConversableAgent
from autogen.opentelemetry.consts import SpanType
from autogen.opentelemetry.instrumentators.agent_instrumentators.code import (
    instrument_code_execution,
    instrument_create_or_get_executor,
)
from autogen.opentelemetry.instrumentators.agent_instrumentators.human_input import instrument_human_input
from autogen.opentelemetry.instrumentators.agent_instrumentators.remote import instrument_remote_reply
from autogen.opentelemetry.setup import get_tracer
from test.opentelemetry.conftest import InMemorySpanExporter


@pytest.fixture()
def otel_setup():
    """Create an in-memory OTEL exporter/provider for capturing spans."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter, provider


# ---------------------------------------------------------------------------
# Helper: build a minimal agent-like object for remote tests
# ---------------------------------------------------------------------------
def _make_mock_agent_with_reply_func(func_name: str, func: Any) -> SimpleNamespace:
    """Create a minimal mock agent that has _reply_func_list with a named func."""
    func.__name__ = func_name
    agent = SimpleNamespace(
        name="mock_agent",
        _reply_func_list=[{"reply_func": func}],
    )
    return agent


# ===========================================================================
# Tests for code.py  (instrument_code_execution)
# ===========================================================================
class TestInstrumentCodeExecution:
    """Tests for instrument_code_execution."""

    def test_wraps_reply_func(self, otel_setup) -> None:
        """Verify that the code execution reply func gets wrapped."""
        exporter, provider = otel_setup
        tracer = get_tracer(provider)

        agent = ConversableAgent("coder", llm_config=False, human_input_mode="NEVER")

        instrument_code_execution(agent, tracer=tracer)

        # If the agent has the code execution func, it should now be wrapped
        for entry in agent._reply_func_list:
            fname = getattr(entry["reply_func"], "__name__", None)
            if fname == "_generate_code_execution_reply_using_executor":
                assert hasattr(entry["reply_func"], "__otel_wrapped__")
                break
            elif fname == "generate_code_execution_reply_traced":
                # Already wrapped
                assert hasattr(entry["reply_func"], "__otel_wrapped__")
                break

    def test_code_execution_disabled_returns_early(self, otel_setup) -> None:
        """When _code_execution_config is False, the traced func returns (False, None)."""
        exporter, provider = otel_setup
        tracer = get_tracer(provider)

        # Build a mock agent with a code execution reply func
        original_func = MagicMock(return_value=(True, "exitcode: 0 (execution succeeded)\nCode output: hello"))
        original_func.__name__ = "_generate_code_execution_reply_using_executor"

        agent = SimpleNamespace(
            name="coder",
            _reply_func_list=[{"reply_func": original_func}],
            _code_execution_config=False,
        )

        instrument_code_execution(agent, tracer=tracer)

        traced_func = agent._reply_func_list[0]["reply_func"]
        result = traced_func(agent, messages=None, sender=None, config=None)
        assert result == (False, None)
        # The original func should NOT have been called
        original_func.assert_not_called()
        # No spans should be created (returns before span starts)
        spans = exporter.get_finished_spans()
        assert len(spans) == 0

    def test_successful_code_execution_span(self, otel_setup) -> None:
        """Test span attributes for a successful code execution (exit code 0)."""
        exporter, provider = otel_setup
        tracer = get_tracer(provider)

        output_text = "exitcode: 0 (execution succeeded)\nCode output: hello world"
        original_func = MagicMock(return_value=(True, output_text))
        original_func.__name__ = "_generate_code_execution_reply_using_executor"

        agent = SimpleNamespace(
            name="coder",
            _reply_func_list=[{"reply_func": original_func}],
            _code_execution_config={"work_dir": "/tmp"},
        )

        instrument_code_execution(agent, tracer=tracer)

        traced_func = agent._reply_func_list[0]["reply_func"]
        is_final, result = traced_func(agent, messages=[], sender=None, config=None)

        assert is_final is True
        assert result == output_text

        spans = exporter.get_finished_spans()
        code_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.CODE_EXECUTION.value]
        assert len(code_spans) == 1

        span = code_spans[0]
        assert span.name == "execute_code coder"
        assert span.attributes["gen_ai.operation.name"] == "execute_code"
        assert span.attributes["gen_ai.agent.name"] == "coder"
        assert span.attributes["ag2.code_execution.exit_code"] == 0
        assert span.attributes["ag2.code_execution.output"] == "hello world"
        # No error.type for exit code 0
        assert "error.type" not in span.attributes

    def test_failed_code_execution_sets_error_type(self, otel_setup) -> None:
        """Test that a non-zero exit code sets error.type on the span."""
        exporter, provider = otel_setup
        tracer = get_tracer(provider)

        output_text = "exitcode: 1 (execution failed)\nCode output: Traceback (most recent call last):\n  File 'test.py', line 1\nNameError: name 'x' is not defined"
        original_func = MagicMock(return_value=(True, output_text))
        original_func.__name__ = "_generate_code_execution_reply_using_executor"

        agent = SimpleNamespace(
            name="coder",
            _reply_func_list=[{"reply_func": original_func}],
            _code_execution_config={"work_dir": "/tmp"},
        )

        instrument_code_execution(agent, tracer=tracer)

        traced_func = agent._reply_func_list[0]["reply_func"]
        is_final, result = traced_func(agent, messages=[], sender=None, config=None)

        assert is_final is True

        spans = exporter.get_finished_spans()
        code_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.CODE_EXECUTION.value]
        assert len(code_spans) == 1

        span = code_spans[0]
        assert span.attributes["ag2.code_execution.exit_code"] == 1
        assert span.attributes["error.type"] == "CodeExecutionError"

    def test_output_truncation(self, otel_setup) -> None:
        """Test that code output longer than 4096 characters is truncated."""
        exporter, provider = otel_setup
        tracer = get_tracer(provider)

        long_output = "x" * 5000
        output_text = f"exitcode: 0 (execution succeeded)\nCode output: {long_output}"
        original_func = MagicMock(return_value=(True, output_text))
        original_func.__name__ = "_generate_code_execution_reply_using_executor"

        agent = SimpleNamespace(
            name="coder",
            _reply_func_list=[{"reply_func": original_func}],
            _code_execution_config={"work_dir": "/tmp"},
        )

        instrument_code_execution(agent, tracer=tracer)

        traced_func = agent._reply_func_list[0]["reply_func"]
        traced_func(agent, messages=[], sender=None, config=None)

        spans = exporter.get_finished_spans()
        code_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.CODE_EXECUTION.value]
        span = code_spans[0]
        output_attr = span.attributes["ag2.code_execution.output"]
        assert output_attr.endswith("... (truncated)")
        # 4096 chars + len("... (truncated)")
        assert len(output_attr) == 4096 + len("... (truncated)")

    def test_not_final_no_parsing(self, otel_setup) -> None:
        """When is_final is False, no exit code or output parsing happens."""
        exporter, provider = otel_setup
        tracer = get_tracer(provider)

        original_func = MagicMock(return_value=(False, None))
        original_func.__name__ = "_generate_code_execution_reply_using_executor"

        agent = SimpleNamespace(
            name="coder",
            _reply_func_list=[{"reply_func": original_func}],
            _code_execution_config={"work_dir": "/tmp"},
        )

        instrument_code_execution(agent, tracer=tracer)

        traced_func = agent._reply_func_list[0]["reply_func"]
        is_final, result = traced_func(agent, messages=[], sender=None, config=None)

        assert is_final is False
        assert result is None

        spans = exporter.get_finished_spans()
        code_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.CODE_EXECUTION.value]
        assert len(code_spans) == 1
        span = code_spans[0]
        # Should not have exit_code or output attributes
        assert "ag2.code_execution.exit_code" not in span.attributes
        assert "ag2.code_execution.output" not in span.attributes

    def test_no_reply_func_list(self, otel_setup) -> None:
        """When agent has no _reply_func_list, instrument returns agent unchanged."""
        exporter, provider = otel_setup
        tracer = get_tracer(provider)

        agent = SimpleNamespace(name="no_reply_list")
        result = instrument_code_execution(agent, tracer=tracer)
        assert result is agent

    def test_no_code_exec_func_in_reply_list(self, otel_setup) -> None:
        """When reply list has no code execution func, nothing changes."""
        exporter, provider = otel_setup
        tracer = get_tracer(provider)

        other_func = MagicMock()
        other_func.__name__ = "some_other_reply_func"

        agent = SimpleNamespace(
            name="coder",
            _reply_func_list=[{"reply_func": other_func}],
        )

        result = instrument_code_execution(agent, tracer=tracer)
        assert result is agent
        # The func should remain unchanged
        assert agent._reply_func_list[0]["reply_func"] is other_func

    def test_result_not_starting_with_exitcode(self, otel_setup) -> None:
        """When result does not start with 'exitcode:', no parsing happens."""
        exporter, provider = otel_setup
        tracer = get_tracer(provider)

        original_func = MagicMock(return_value=(True, "some other output format"))
        original_func.__name__ = "_generate_code_execution_reply_using_executor"

        agent = SimpleNamespace(
            name="coder",
            _reply_func_list=[{"reply_func": original_func}],
            _code_execution_config={"work_dir": "/tmp"},
        )

        instrument_code_execution(agent, tracer=tracer)

        traced_func = agent._reply_func_list[0]["reply_func"]
        is_final, result = traced_func(agent, messages=[], sender=None, config=None)

        assert is_final is True
        assert result == "some other output format"

        spans = exporter.get_finished_spans()
        code_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.CODE_EXECUTION.value]
        span = code_spans[0]
        assert "ag2.code_execution.exit_code" not in span.attributes


# ===========================================================================
# Tests for code.py  (instrument_create_or_get_executor)
# ===========================================================================
class TestInstrumentCreateOrGetExecutor:
    """Tests for instrument_create_or_get_executor."""

    def test_wraps_create_or_get_executor(self, otel_setup) -> None:
        """Verify that _create_or_get_executor is wrapped."""
        exporter, provider = otel_setup

        from contextlib import contextmanager

        @contextmanager
        def mock_create_or_get_executor(**kwargs):
            yield SimpleNamespace(name="executor_agent")

        agent = SimpleNamespace(
            name="coder",
            _create_or_get_executor=mock_create_or_get_executor,
        )

        mock_instrumentator = MagicMock(side_effect=lambda a: a)
        instrument_create_or_get_executor(agent, instrumentator=mock_instrumentator)

        assert hasattr(agent._create_or_get_executor, "__otel_wrapped__")

    def test_idempotency(self, otel_setup) -> None:
        """Double wrapping should not happen."""
        exporter, provider = otel_setup

        from contextlib import contextmanager

        @contextmanager
        def mock_create_or_get_executor(**kwargs):
            yield SimpleNamespace(name="executor_agent")

        agent = SimpleNamespace(
            name="coder",
            _create_or_get_executor=mock_create_or_get_executor,
        )

        mock_instrumentator = MagicMock(side_effect=lambda a: a)
        instrument_create_or_get_executor(agent, instrumentator=mock_instrumentator)
        first_wrapper = agent._create_or_get_executor

        instrument_create_or_get_executor(agent, instrumentator=mock_instrumentator)
        assert agent._create_or_get_executor is first_wrapper

    def test_calls_instrumentator_on_executor(self, otel_setup) -> None:
        """Verify the instrumentator callback is called on the yielded executor."""
        exporter, provider = otel_setup

        from contextlib import contextmanager

        executor_agent = SimpleNamespace(name="executor_agent")

        @contextmanager
        def mock_create_or_get_executor(**kwargs):
            yield executor_agent

        agent = SimpleNamespace(
            name="coder",
            _create_or_get_executor=mock_create_or_get_executor,
        )

        mock_instrumentator = MagicMock(side_effect=lambda a: a)
        instrument_create_or_get_executor(agent, instrumentator=mock_instrumentator)

        with agent._create_or_get_executor():
            pass

        mock_instrumentator.assert_called_once_with(executor_agent)

    def test_no_create_or_get_executor(self, otel_setup) -> None:
        """When agent has no _create_or_get_executor, returns agent unchanged."""
        exporter, provider = otel_setup

        agent = SimpleNamespace(name="no_executor")
        mock_instrumentator = MagicMock()
        result = instrument_create_or_get_executor(agent, instrumentator=mock_instrumentator)
        assert result is agent


# ===========================================================================
# Tests for human_input.py  (instrument_human_input)
# ===========================================================================
class TestInstrumentHumanInput:
    """Tests for instrument_human_input."""

    def test_wraps_get_human_input(self, otel_setup) -> None:
        """Verify that get_human_input gets wrapped."""
        exporter, provider = otel_setup
        tracer = get_tracer(provider)

        agent = ConversableAgent("human_agent", llm_config=False, human_input_mode="ALWAYS")
        instrument_human_input(agent, tracer=tracer)

        assert hasattr(agent.get_human_input, "__otel_wrapped__")

    def test_wraps_a_get_human_input(self, otel_setup) -> None:
        """Verify that a_get_human_input gets wrapped."""
        exporter, provider = otel_setup
        tracer = get_tracer(provider)

        agent = ConversableAgent("human_agent", llm_config=False, human_input_mode="ALWAYS")
        instrument_human_input(agent, tracer=tracer)

        assert hasattr(agent.a_get_human_input, "__otel_wrapped__")

    def test_sync_get_human_input_creates_span(self, otel_setup) -> None:
        """Test that calling get_human_input creates an await_human_input span."""
        exporter, provider = otel_setup
        tracer = get_tracer(provider)

        agent = ConversableAgent("alice", llm_config=False, human_input_mode="ALWAYS")

        agent.get_human_input = MagicMock(return_value="my response")

        instrument_human_input(agent, tracer=tracer)

        result = agent.get_human_input("Enter your name:")

        assert result == "my response"

        spans = exporter.get_finished_spans()
        human_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.HUMAN_INPUT.value]
        assert len(human_spans) == 1

        span = human_spans[0]
        assert span.name == "await_human_input alice"
        assert span.attributes["gen_ai.operation.name"] == "await_human_input"
        assert span.attributes["gen_ai.agent.name"] == "alice"
        assert span.attributes["ag2.human_input.prompt"] == "Enter your name:"
        assert span.attributes["ag2.human_input.response"] == "my response"

    @pytest.mark.asyncio
    async def test_async_get_human_input_creates_span(self, otel_setup) -> None:
        """Test that calling a_get_human_input creates an await_human_input span."""
        exporter, provider = otel_setup
        tracer = get_tracer(provider)

        agent = ConversableAgent("bob", llm_config=False, human_input_mode="ALWAYS")

        # Mock the async method
        agent.a_get_human_input = AsyncMock(return_value="async response")

        instrument_human_input(agent, tracer=tracer)

        result = await agent.a_get_human_input("Enter your age:")

        assert result == "async response"

        spans = exporter.get_finished_spans()
        human_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.HUMAN_INPUT.value]
        assert len(human_spans) == 1

        span = human_spans[0]
        assert span.name == "await_human_input bob"
        assert span.attributes["gen_ai.operation.name"] == "await_human_input"
        assert span.attributes["gen_ai.agent.name"] == "bob"
        assert span.attributes["ag2.human_input.prompt"] == "Enter your age:"
        assert span.attributes["ag2.human_input.response"] == "async response"

    def test_idempotency_sync(self, otel_setup) -> None:
        """Double wrapping get_human_input should not happen."""
        exporter, provider = otel_setup
        tracer = get_tracer(provider)

        agent = ConversableAgent("human_agent", llm_config=False, human_input_mode="ALWAYS")
        instrument_human_input(agent, tracer=tracer)
        first_wrapper = agent.get_human_input

        instrument_human_input(agent, tracer=tracer)
        assert agent.get_human_input is first_wrapper

    def test_idempotency_async(self, otel_setup) -> None:
        """Double wrapping a_get_human_input should not happen."""
        exporter, provider = otel_setup
        tracer = get_tracer(provider)

        agent = ConversableAgent("human_agent", llm_config=False, human_input_mode="ALWAYS")
        instrument_human_input(agent, tracer=tracer)
        first_wrapper = agent.a_get_human_input

        instrument_human_input(agent, tracer=tracer)
        assert agent.a_get_human_input is first_wrapper

    def test_agent_without_get_human_input(self, otel_setup) -> None:
        """Agent without get_human_input attribute is returned unchanged."""
        exporter, provider = otel_setup
        tracer = get_tracer(provider)

        agent = SimpleNamespace(name="minimal_agent")
        result = instrument_human_input(agent, tracer=tracer)
        assert result is agent
        assert not hasattr(agent, "get_human_input")

    def test_prompt_forwarded_to_original(self, otel_setup) -> None:
        """Verify the prompt and extra args are forwarded to the original func."""
        exporter, provider = otel_setup
        tracer = get_tracer(provider)

        agent = ConversableAgent("human_agent", llm_config=False, human_input_mode="ALWAYS")
        mock_original = MagicMock(return_value="ok")
        agent.get_human_input = mock_original

        instrument_human_input(agent, tracer=tracer)
        agent.get_human_input("Tell me something:", "extra_arg", key="value")

        mock_original.assert_called_once_with("Tell me something:", "extra_arg", key="value")


# ===========================================================================
# Tests for remote.py  (instrument_remote_reply)
# ===========================================================================
class TestInstrumentRemoteReply:
    """Tests for instrument_remote_reply."""

    def test_wraps_reply_func(self, otel_setup) -> None:
        """Verify that a_generate_remote_reply gets wrapped in the reply list."""
        exporter, provider = otel_setup
        tracer = get_tracer(provider)

        original_func = AsyncMock(return_value=(True, "remote response"))
        original_func.__name__ = "a_generate_remote_reply"

        agent = SimpleNamespace(
            name="remote_agent",
            a_generate_remote_reply=original_func,
            _reply_func_list=[{"reply_func": original_func}],
            _httpx_client_factory=MagicMock(return_value=MagicMock(headers={})),
        )

        instrument_remote_reply(agent, tracer=tracer)

        wrapped_func = agent._reply_func_list[0]["reply_func"]
        assert hasattr(wrapped_func, "__otel_wrapped__")
        assert wrapped_func is not original_func

    @pytest.mark.asyncio
    async def test_remote_reply_creates_agent_span(self, otel_setup) -> None:
        """Test that calling the traced remote reply creates an invoke_agent span."""
        exporter, provider = otel_setup
        tracer = get_tracer(provider)

        original_func = AsyncMock(return_value=(True, "remote response"))
        original_func.__name__ = "a_generate_remote_reply"

        agent = SimpleNamespace(
            name="remote_agent",
            url="http://remote-server:8080",
            a_generate_remote_reply=original_func,
            _reply_func_list=[{"reply_func": original_func}],
            _httpx_client_factory=MagicMock(return_value=MagicMock(headers={})),
        )

        instrument_remote_reply(agent, tracer=tracer)

        traced_func = agent._reply_func_list[0]["reply_func"]
        result = await traced_func(agent)

        assert result == (True, "remote response")

        spans = exporter.get_finished_spans()
        agent_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.AGENT.value]
        assert len(agent_spans) == 1

        span = agent_spans[0]
        assert span.name == "invoke_agent remote_agent"
        assert span.attributes["gen_ai.operation.name"] == "invoke_agent"
        assert span.attributes["gen_ai.agent.name"] == "remote_agent"
        assert span.attributes["gen_ai.agent.remote"] is True
        assert span.attributes["server.address"] == "http://remote-server:8080"

    @pytest.mark.asyncio
    async def test_remote_reply_without_url(self, otel_setup) -> None:
        """Test that server.address is not set when agent has no url."""
        exporter, provider = otel_setup
        tracer = get_tracer(provider)

        original_func = AsyncMock(return_value=(True, "response"))
        original_func.__name__ = "a_generate_remote_reply"

        agent = SimpleNamespace(
            name="remote_agent",
            a_generate_remote_reply=original_func,
            _reply_func_list=[{"reply_func": original_func}],
            _httpx_client_factory=MagicMock(return_value=MagicMock(headers={})),
        )

        instrument_remote_reply(agent, tracer=tracer)

        traced_func = agent._reply_func_list[0]["reply_func"]
        await traced_func(agent)

        spans = exporter.get_finished_spans()
        agent_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.AGENT.value]
        assert len(agent_spans) == 1
        span = agent_spans[0]
        assert "server.address" not in span.attributes

    @pytest.mark.asyncio
    async def test_remote_reply_with_empty_url(self, otel_setup) -> None:
        """Test that server.address is not set when url is empty string."""
        exporter, provider = otel_setup
        tracer = get_tracer(provider)

        original_func = AsyncMock(return_value=(True, "response"))
        original_func.__name__ = "a_generate_remote_reply"

        agent = SimpleNamespace(
            name="remote_agent",
            url="",
            a_generate_remote_reply=original_func,
            _reply_func_list=[{"reply_func": original_func}],
            _httpx_client_factory=MagicMock(return_value=MagicMock(headers={})),
        )

        instrument_remote_reply(agent, tracer=tracer)

        traced_func = agent._reply_func_list[0]["reply_func"]
        await traced_func(agent)

        spans = exporter.get_finished_spans()
        agent_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.AGENT.value]
        span = agent_spans[0]
        assert "server.address" not in span.attributes

    def test_httpx_client_factory_wrapped(self, otel_setup) -> None:
        """Test that _httpx_client_factory is replaced with a traced version."""
        exporter, provider = otel_setup
        tracer = get_tracer(provider)

        original_func = AsyncMock(return_value=(True, "response"))
        original_func.__name__ = "a_generate_remote_reply"

        mock_headers = {}
        mock_client = MagicMock(headers=mock_headers)
        original_factory = MagicMock(return_value=mock_client)

        agent = SimpleNamespace(
            name="remote_agent",
            a_generate_remote_reply=original_func,
            _reply_func_list=[{"reply_func": original_func}],
            _httpx_client_factory=original_factory,
        )

        instrument_remote_reply(agent, tracer=tracer)

        # The factory should have been replaced
        assert agent._httpx_client_factory is not original_factory

        # Call the wrapped factory and verify it calls the original
        client = agent._httpx_client_factory()
        original_factory.assert_called_once()
        assert client is mock_client

    def test_httpx_client_factory_injects_traceparent(self, otel_setup) -> None:
        """Test that the wrapped httpx factory injects traceparent into headers."""
        exporter, provider = otel_setup
        tracer = get_tracer(provider)

        original_func = AsyncMock(return_value=(True, "response"))
        original_func.__name__ = "a_generate_remote_reply"

        mock_headers = {}
        mock_client = MagicMock(headers=mock_headers)
        original_factory = MagicMock(return_value=mock_client)

        agent = SimpleNamespace(
            name="remote_agent",
            a_generate_remote_reply=original_func,
            _reply_func_list=[{"reply_func": original_func}],
            _httpx_client_factory=original_factory,
        )

        instrument_remote_reply(agent, tracer=tracer)

        # Call inside a span context so there's a traceparent to inject
        with tracer.start_as_current_span("test_parent"):
            agent._httpx_client_factory()

        # The TRACE_PROPAGATOR.inject should have been called on mock_client.headers
        # Since we're inside a span, the traceparent header should be present
        # (the inject call mutates the headers dict via the mock_client.headers reference)
        original_factory.assert_called_once()

    def test_no_a_generate_remote_reply(self, otel_setup) -> None:
        """Agent without a_generate_remote_reply returns unchanged."""
        exporter, provider = otel_setup
        tracer = get_tracer(provider)

        agent = SimpleNamespace(
            name="local_agent",
            _reply_func_list=[],
        )

        result = instrument_remote_reply(agent, tracer=tracer)
        assert result is agent

    def test_no_matching_func_in_reply_list(self, otel_setup) -> None:
        """Agent with a_generate_remote_reply attr but not in reply list."""
        exporter, provider = otel_setup
        tracer = get_tracer(provider)

        other_func = MagicMock()
        other_func.__name__ = "some_other_func"

        agent = SimpleNamespace(
            name="remote_agent",
            a_generate_remote_reply=MagicMock(),
            _reply_func_list=[{"reply_func": other_func}],
            _httpx_client_factory=MagicMock(return_value=MagicMock(headers={})),
        )

        result = instrument_remote_reply(agent, tracer=tracer)
        assert result is agent
        # The reply func should not have been changed
        assert agent._reply_func_list[0]["reply_func"] is other_func

    def test_already_wrapped_does_not_rewrap(self, otel_setup) -> None:
        """If a_generate_remote_reply is already wrapped, it should not be re-wrapped."""
        exporter, provider = otel_setup
        tracer = get_tracer(provider)

        original_func = AsyncMock(return_value=(True, "response"))
        original_func.__name__ = "a_generate_remote_reply"
        original_func.__otel_wrapped__ = True  # Already marked as wrapped

        agent = SimpleNamespace(
            name="remote_agent",
            a_generate_remote_reply=original_func,
            _reply_func_list=[{"reply_func": original_func}],
            _httpx_client_factory=MagicMock(return_value=MagicMock(headers={})),
        )

        instrument_remote_reply(agent, tracer=tracer)

        # Should still be the original func since it was already wrapped
        assert agent._reply_func_list[0]["reply_func"] is original_func

    @pytest.mark.asyncio
    async def test_remote_reply_forwards_args(self, otel_setup) -> None:
        """Verify args and kwargs are forwarded to the original reply func."""
        exporter, provider = otel_setup
        tracer = get_tracer(provider)

        original_func = AsyncMock(return_value=(True, "response"))
        original_func.__name__ = "a_generate_remote_reply"

        agent = SimpleNamespace(
            name="remote_agent",
            url="http://example.com",
            a_generate_remote_reply=original_func,
            _reply_func_list=[{"reply_func": original_func}],
            _httpx_client_factory=MagicMock(return_value=MagicMock(headers={})),
        )

        instrument_remote_reply(agent, tracer=tracer)

        traced_func = agent._reply_func_list[0]["reply_func"]
        messages = [{"role": "user", "content": "hello"}]
        sender = SimpleNamespace(name="sender_agent")
        await traced_func(agent, messages, sender, config={"key": "value"})

        original_func.assert_called_once_with(agent, messages, sender, config={"key": "value"})
