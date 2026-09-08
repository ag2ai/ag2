# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
import pytest

from ag2 import Agent, MemoryStream
from ag2.events import (
    BaseEvent,
    BuiltinToolCallEvent,
    BuiltinToolResultEvent,
    ModelMessage,
    ModelMessageChunk,
    ModelReasoning,
    ToolCallEvent,
)
from ag2.middleware import RetryMiddleware
from ag2.testing import TestConfig, TrackingConfig, Turn
from ag2.tools import ToolResult, tool


class TransientError(Exception):
    pass


class PermanentError(Exception):
    pass


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def retrying_agent(*script: Turn, max_retries: int = 3) -> tuple[Agent, TrackingConfig]:
    """An agent that retries ``TransientError`` and nothing else.

    The returned config counts LLM calls, failed attempts included, so a test can
    say how many times the model was asked.
    """
    config = TrackingConfig(TestConfig(*script))
    agent = Agent(
        "retrying",
        config=config,
        tools=[add],
        middleware=[RetryMiddleware(max_retries=max_retries, retry_on=(TransientError,))],
    )
    return agent, config


@pytest.mark.asyncio()
async def test_a_call_that_succeeds_is_made_once() -> None:
    agent, config = retrying_agent("Hello!")

    reply = await agent.ask("Hi!")

    assert reply.body == "Hello!"
    assert config.mock.call_count == 1


@pytest.mark.asyncio()
async def test_a_failing_call_is_retried_until_it_succeeds() -> None:
    agent, config = retrying_agent(
        TransientError("transient failure 1"),
        TransientError("transient failure 2"),
        "Hello!",
    )

    reply = await agent.ask("Hi!")

    assert reply.body == "Hello!"
    assert config.mock.call_count == 3


@pytest.mark.asyncio()
async def test_the_error_surfaces_once_the_retries_run_out() -> None:
    # ``max_retries`` counts the retries, so the model is asked one more time
    # than that before the failure is allowed through.
    agent, config = retrying_agent(*(TransientError("still failing") for _ in range(4)), max_retries=3)

    with pytest.raises(TransientError, match="still failing"):
        await agent.ask("Hi!")

    assert config.mock.call_count == 4


@pytest.mark.asyncio()
async def test_an_error_outside_retry_on_is_not_retried() -> None:
    agent, config = retrying_agent(PermanentError("do not retry"), "never reached")

    with pytest.raises(PermanentError, match="do not retry"):
        await agent.ask("Hi!")

    assert config.mock.call_count == 1


@pytest.mark.parametrize(
    "published",
    [
        ModelMessageChunk("stale "),
        ModelReasoning("half a thought"),
        ModelMessage("stale"),
        BuiltinToolCallEvent("web_search"),
        BuiltinToolResultEvent(parent_id="call-1", result=ToolResult("stale")),
    ],
    ids=["chunk", "reasoning", "message", "builtin_call", "builtin_result"],
)
@pytest.mark.asyncio()
async def test_a_call_that_published_output_is_not_retried(published: BaseEvent) -> None:
    """A retry cannot take back what consumers already have, so it must not happen.

    The cases are what a provider client publishes mid-call — streamed content,
    the message, server-side tool activity — but the middleware names no types:
    anything reaching the stream inside the call counts.
    """
    agent, config = retrying_agent(published, TransientError("stream disconnected"), "would be a second copy")

    seen: list[BaseEvent] = []
    stream = MemoryStream()

    async def collect(event: BaseEvent) -> None:
        seen.append(event)

    stream.subscribe(collect, sync_to_thread=False)

    with pytest.raises(TransientError, match="stream disconnected"):
        await agent.ask("Hi!", stream=stream)

    assert config.mock.call_count == 1
    # Consumers keep the partial output — exactly once. A retry would have made it two.
    assert seen.count(published) == 1


@pytest.mark.asyncio()
async def test_output_from_an_earlier_turn_does_not_disarm_the_next_one() -> None:
    """The guard is scoped to one call, and the loop's own events stay outside it.

    Turn one streams a chunk and asks for a tool; the loop then publishes the
    tool call and its result. None of that is turn two's output, so turn two is
    still free to retry. Were the scope wider — or were the loop to publish from
    inside the middleware chain — every retry after a tool call would silently
    stop happening.
    """
    agent, config = retrying_agent(
        ModelMessageChunk("Let me add those"),  # turn 1 streams...
        ToolCallEvent(name="add", arguments='{"a": 1, "b": 2}'),  # ...then calls the tool
        TransientError("transient failure"),  # turn 2 fails...
        "The answer is 3.",  # ...and its retry succeeds
    )

    reply = await agent.ask("What is 1 + 2?")

    assert reply.body == "The answer is 3."
    assert config.mock.call_count == 3
