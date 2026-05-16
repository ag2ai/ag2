# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for AgentReply.usage / AgentReply.total_usage() and Usage.__add__."""

import pytest

from autogen.beta import Agent
from autogen.beta.events import (
    ModelMessage,
    ModelResponse,
    ToolCallEvent,
    ToolCallsEvent,
    Usage,
)
from autogen.beta.testing import TestConfig
from autogen.beta.tools import tool

# ---------------------------------------------------------------------------
# Usage.__add__
# ---------------------------------------------------------------------------


def test_usage_add_both_populated() -> None:
    a = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    b = Usage(prompt_tokens=3, completion_tokens=2, total_tokens=5)
    c = a + b
    assert c.prompt_tokens == 13
    assert c.completion_tokens == 7
    assert c.total_tokens == 20


def test_usage_add_none_stays_none_when_both_none() -> None:
    a = Usage(prompt_tokens=None)
    b = Usage(prompt_tokens=None)
    assert (a + b).prompt_tokens is None


def test_usage_add_none_plus_value() -> None:
    a = Usage(prompt_tokens=None)
    b = Usage(prompt_tokens=7)
    assert (a + b).prompt_tokens == 7
    assert (b + a).prompt_tokens == 7


def test_usage_add_cache_tokens() -> None:
    a = Usage(cache_read_input_tokens=100, cache_creation_input_tokens=50)
    b = Usage(cache_read_input_tokens=20, cache_creation_input_tokens=10)
    c = a + b
    assert c.cache_read_input_tokens == 120
    assert c.cache_creation_input_tokens == 60


# ---------------------------------------------------------------------------
# AgentReply.usage — single turn, no tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reply_usage_single_turn() -> None:
    agent = Agent(
        "assistant",
        config=TestConfig(
            ModelResponse(
                message=ModelMessage("Hi!"),
                usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )
        ),
    )
    reply = await agent.ask("Hello")
    assert reply.usage.prompt_tokens == 10
    assert reply.usage.completion_tokens == 5
    assert reply.usage.total_tokens == 15


@pytest.mark.asyncio
async def test_reply_usage_no_usage_info() -> None:
    agent = Agent(
        "assistant",
        config=TestConfig(ModelResponse(message=ModelMessage("Hi!"))),
    )
    reply = await agent.ask("Hello")
    # Usage is always a Usage object (never None); empty when provider omits it
    assert isinstance(reply.usage, Usage)
    assert not reply.usage


# ---------------------------------------------------------------------------
# AgentReply.total_usage — accumulates across tool-calling loops
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_total_usage_accumulates_across_tool_calls() -> None:
    @tool
    async def add(a: int, b: int) -> int:
        """Return a + b."""
        return a + b

    # Two LLM calls: first returns a tool call, second returns final reply
    agent = Agent(
        "assistant",
        config=TestConfig(
            ModelResponse(
                tool_calls=ToolCallsEvent(calls=[ToolCallEvent(name="add", arguments='{"a": 1, "b": 2}')]),
                usage=Usage(prompt_tokens=8, completion_tokens=3, total_tokens=11),
            ),
            ModelResponse(
                message=ModelMessage("The answer is 3."),
                usage=Usage(prompt_tokens=20, completion_tokens=6, total_tokens=26),
            ),
        ),
        tools=[add],
    )
    reply = await agent.ask("What is 1 + 2?")

    # .usage is only the final call
    assert reply.usage.prompt_tokens == 20

    # .total_usage() sums both calls
    total = await reply.total_usage()
    assert total.prompt_tokens == 28
    assert total.completion_tokens == 9
    assert total.total_tokens == 37


@pytest.mark.asyncio
async def test_total_usage_single_turn_matches_usage() -> None:
    agent = Agent(
        "assistant",
        config=TestConfig(
            ModelResponse(
                message=ModelMessage("Hi!"),
                usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )
        ),
    )
    reply = await agent.ask("Hello")
    total = await reply.total_usage()
    # Single LLM call: total_usage == usage
    assert total.prompt_tokens == reply.usage.prompt_tokens
    assert total.completion_tokens == reply.usage.completion_tokens
    assert total.total_tokens == reply.usage.total_tokens
