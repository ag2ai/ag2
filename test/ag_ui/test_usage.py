# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Token usage as an AG-UI client sees it on the run-terminating events.

Every assertion is made on the events a client actually receives from ``dispatch``,
parsed back into the protocol's own models — nothing reaches for the mapping helper.
A run spanning several provider/model pairs is constructible because the test client
returns each supplied ``ModelResponse`` verbatim, and the agent emits that response's
own ``model``, ``provider`` and ``usage`` onto the stream.
"""

from typing import Any

import pytest
from ag_ui.core import RunErrorEvent, RunFinishedEvent, TokenUsage, UserMessage

from ag2 import Agent
from ag2.ag_ui import AGUIStream
from ag2.events import ModelMessage, ModelResponse, ToolCallEvent, ToolCallsEvent, Usage
from ag2.testing import TestConfig
from ag2.tools import tool

from .utils import collect_events, create_run_input, leaf_exceptions

pytestmark = pytest.mark.asyncio


async def _frames(agent: Agent, *, into: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """The SSE frames one run yields, decoded but not parsed."""
    run_input = create_run_input(UserMessage(id="msg_1", content="go"))
    return await collect_events(AGUIStream(agent), run_input, into=into)


async def _finished(agent: Agent) -> RunFinishedEvent:
    """The terminating event of a completed run.

    Taken as the last frame and parsed by the class the implementation sends. The class
    rejects a frame of any other type, so this also pins that the run really did end on
    ``RUN_FINISHED`` — no separate search by type is needed.
    """
    return RunFinishedEvent.model_validate((await _frames(agent))[-1])


async def _run_error(agent: Agent) -> RunErrorEvent:
    """The terminating event of a failing run, emitted before ``dispatch`` re-raises."""
    frames: list[dict[str, Any]] = []
    with pytest.raises(Exception):
        await _frames(agent, into=frames)
    return RunErrorEvent.model_validate(frames[-1])


def _exploding_agent(usage: Usage | None = None) -> Agent:
    @tool
    def explode() -> str:
        """Always fails."""
        raise RuntimeError("downstream is down")

    call = ToolCallsEvent(calls=[ToolCallEvent(name="explode", arguments="{}")])
    response = (
        ModelResponse(tool_calls=call, usage=usage, model="claude-sonnet-4", provider="anthropic")
        if usage
        else ModelResponse(tool_calls=call)
    )
    return Agent("test_agent", config=TestConfig(response), tools=[explode])


class TestCompletedRun:
    async def test_reports_input_output_and_total(self) -> None:
        agent = Agent(
            "test_agent",
            config=TestConfig(
                ModelResponse(
                    ModelMessage("done"),
                    usage=Usage(prompt_tokens=120, completion_tokens=48, total_tokens=168),
                    model="claude-sonnet-4",
                    provider="anthropic",
                ),
            ),
        )

        assert (await _finished(agent)).usage == [
            TokenUsage(
                provider="anthropic",
                model="claude-sonnet-4",
                input_tokens=120,
                output_tokens=48,
                total_tokens=168,
            )
        ]

    async def test_sums_every_model_call_not_just_the_last(self) -> None:
        @tool
        def lookup() -> str:
            """Look something up."""
            return "42"

        agent = Agent(
            "test_agent",
            config=TestConfig(
                ModelResponse(
                    tool_calls=ToolCallsEvent(calls=[ToolCallEvent(name="lookup", arguments="{}")]),
                    usage=Usage(prompt_tokens=100, completion_tokens=10, total_tokens=110),
                    model="gpt-5",
                    provider="openai",
                ),
                ModelResponse(
                    ModelMessage("it is 42"),
                    usage=Usage(prompt_tokens=40, completion_tokens=4, total_tokens=44),
                    model="gpt-5",
                    provider="openai",
                ),
            ),
            tools=[lookup],
        )

        assert (await _finished(agent)).usage == [
            TokenUsage(provider="openai", model="gpt-5", input_tokens=140, output_tokens=14, total_tokens=154)
        ]

    async def test_one_entry_per_pair_in_order_of_first_appearance(self) -> None:
        @tool
        def handoff() -> str:
            """Hand off to the other model."""
            return "ok"

        agent = Agent(
            "test_agent",
            config=TestConfig(
                ModelResponse(
                    tool_calls=ToolCallsEvent(calls=[ToolCallEvent(name="handoff", arguments="{}")]),
                    usage=Usage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
                    model="claude-sonnet-4",
                    provider="anthropic",
                ),
                ModelResponse(
                    ModelMessage("done"),
                    usage=Usage(prompt_tokens=7, completion_tokens=3, total_tokens=10),
                    model="gpt-5",
                    provider="openai",
                ),
            ),
            tools=[handoff],
        )

        assert (await _finished(agent)).usage == [
            TokenUsage(
                provider="anthropic", model="claude-sonnet-4", input_tokens=10, output_tokens=2, total_tokens=12
            ),
            TokenUsage(provider="openai", model="gpt-5", input_tokens=7, output_tokens=3, total_tokens=10),
        ]

    async def test_reports_reasoning_and_cached_input_when_supplied(self) -> None:
        agent = Agent(
            "test_agent",
            config=TestConfig(
                ModelResponse(
                    ModelMessage("done"),
                    usage=Usage(
                        prompt_tokens=100,
                        completion_tokens=30,
                        total_tokens=130,
                        thinking_tokens=18,
                        cache_read_input_tokens=64,
                    ),
                    model="gpt-5",
                    provider="openai",
                ),
            ),
        )

        assert (await _finished(agent)).usage == [
            TokenUsage(
                provider="openai",
                model="gpt-5",
                input_tokens=100,
                output_tokens=30,
                total_tokens=130,
                reasoning_tokens=18,
                cached_input_tokens=64,
            )
        ]

    async def test_omits_fields_the_provider_did_not_report(self) -> None:
        """Absence, never a zero or a derived figure standing in for an unmeasured value.

        The whole-object comparison is what pins it: a total derived from 10 + 4 would
        show up here as ``total_tokens=14``.
        """
        agent = Agent(
            "test_agent",
            config=TestConfig(
                ModelResponse(
                    ModelMessage("done"),
                    usage=Usage(prompt_tokens=10, completion_tokens=4),
                    model="claude-sonnet-4",
                    provider="anthropic",
                ),
            ),
        )

        assert (await _finished(agent)).usage == [
            TokenUsage(provider="anthropic", model="claude-sonnet-4", input_tokens=10, output_tokens=4)
        ]

    async def test_unreported_fields_are_absent_on_the_wire_not_null(self) -> None:
        """The one claim the parsed model cannot carry.

        Parsing puts an omitted field back as ``None``, so absence and an explicit
        ``null`` are indistinguishable once decoded — this asserts on the raw frame.
        """
        agent = Agent(
            "test_agent",
            config=TestConfig(
                ModelResponse(
                    ModelMessage("done"),
                    usage=Usage(prompt_tokens=10, completion_tokens=4),
                    model="claude-sonnet-4",
                    provider="anthropic",
                ),
            ),
        )
        frames = await _frames(agent)

        [entry] = frames[-1]["usage"]
        assert entry == {
            "provider": "anthropic",
            "model": "claude-sonnet-4",
            "inputTokens": 10,
            "outputTokens": 4,
        }

    async def test_cache_write_tokens_appear_nowhere(self) -> None:
        """Cache-write is not cached-input, and is not folded into input either."""
        agent = Agent(
            "test_agent",
            config=TestConfig(
                ModelResponse(
                    ModelMessage("done"),
                    usage=Usage(
                        prompt_tokens=10,
                        completion_tokens=4,
                        cache_creation_input_tokens=512,
                        cache_read_input_tokens=8,
                    ),
                    model="claude-sonnet-4",
                    provider="anthropic",
                ),
            ),
        )

        assert (await _finished(agent)).usage == [
            TokenUsage(
                provider="anthropic",
                model="claude-sonnet-4",
                input_tokens=10,
                output_tokens=4,
                cached_input_tokens=8,
            )
        ]

    async def test_a_run_that_spent_nothing_omits_usage(self) -> None:
        agent = Agent("test_agent", config=TestConfig("hello"))

        assert (await _finished(agent)).usage is None

    async def test_includes_spend_from_a_delegated_subagent(self) -> None:
        worker = Agent(
            "worker",
            config=TestConfig(
                ModelResponse(
                    ModelMessage("researched"),
                    usage=Usage(prompt_tokens=200, completion_tokens=60, total_tokens=260),
                    model="claude-haiku-4",
                    provider="anthropic",
                ),
            ),
        )
        parent = Agent(
            "test_agent",
            config=TestConfig(
                ModelResponse(
                    tool_calls=ToolCallsEvent(
                        calls=[ToolCallEvent(name="task_worker", arguments='{"objective": "go"}')]
                    ),
                    usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                    model="gpt-5",
                    provider="openai",
                ),
                ModelResponse(
                    ModelMessage("summarised"),
                    usage=Usage(prompt_tokens=20, completion_tokens=8, total_tokens=28),
                    model="gpt-5",
                    provider="openai",
                ),
            ),
            tools=[worker.as_tool(description="Delegate research to the worker.")],
        )

        usage = (await _finished(parent)).usage

        assert usage is not None
        assert sum(entry.input_tokens or 0 for entry in usage) == 230, usage


class TestFailedRun:
    async def test_reports_usage_spent_before_the_failure(self) -> None:
        agent = _exploding_agent(Usage(prompt_tokens=250, completion_tokens=50, total_tokens=300))

        assert (await _run_error(agent)).usage == [
            TokenUsage(
                provider="anthropic",
                model="claude-sonnet-4",
                input_tokens=250,
                output_tokens=50,
                total_tokens=300,
            )
        ]

    async def test_a_failure_before_any_spend_omits_usage(self) -> None:
        assert (await _run_error(_exploding_agent())).usage is None

    async def test_the_original_exception_still_reaches_the_caller(self) -> None:
        agent = _exploding_agent(Usage(prompt_tokens=250, completion_tokens=50, total_tokens=300))

        with pytest.raises(Exception) as exc_info:
            await _frames(agent)

        leaves = leaf_exceptions(exc_info.value)
        assert [type(e) for e in leaves] == [RuntimeError]
        assert str(leaves[0]) == "downstream is down"
