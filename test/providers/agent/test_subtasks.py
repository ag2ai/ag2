# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Delegation smoke: as_tool delegation and persistent_stream. Real LLM calls."""

import pytest

from ag2 import Agent
from ag2.history import MemoryStorage
from ag2.stream import MemoryStream
from ag2.tools.subagents import persistent_stream

pytestmark = pytest.mark.asyncio


async def test_actor_as_tool_delegation(provider_config) -> None:
    """A.as_tool() lets agent B call A as a sibling tool."""
    expert = Agent(
        "math-expert",
        prompt="You only do arithmetic. Reply with just the number.",
        config=provider_config,
    )

    coordinator = Agent(
        "coordinator",
        prompt=(
            "You delegate math problems to the task_math-expert tool. "
            "After receiving the answer, present it as a complete sentence."
        ),
        config=provider_config,
        tools=[expert.as_tool(description="Delegate any arithmetic to the math-expert agent.")],
    )

    reply = await coordinator.ask("What is 19 * 23?")
    assert reply.body is not None
    assert "437" in reply.body


async def test_persistent_stream_shares_history(provider_config) -> None:
    """persistent_stream() reuses one stream id + storage across as_tool calls.

    The original test asserted on the parent's reply body, but the parent's
    own conversation history leaks any fact mentioned in the prompt — the
    body assertion would pass even if persistent_stream did nothing. This
    version wraps the factory and asserts on the structural promise:

    1. Both child invocations receive a stream with the **same id**.
    2. Both child streams share the **same storage backend**.

    Both invariants are necessary for cross-call history persistence and
    neither holds under the default per-call StreamFactory.
    """
    captured_streams: list[MemoryStream] = []
    inner = persistent_stream()

    def wrapped_factory(agent, ctx):
        s = inner(agent, ctx)
        captured_streams.append(s)
        return s

    child = Agent(
        "memo",
        prompt="You are a notepad. Reply briefly to whatever the user asks.",
        config=provider_config,
    )

    parent_stream = MemoryStream(storage=MemoryStorage())
    parent = Agent(
        "owner",
        prompt="Use the task_memo tool when the user asks you to.",
        config=provider_config,
        tools=[child.as_tool(description="Notepad agent.", stream=wrapped_factory)],
    )

    reply1 = await parent.ask("Use the task_memo tool to greet Alice.", stream=parent_stream)
    reply2 = await reply1.ask("Use the task_memo tool to greet Bob.")
    assert reply1.body is not None
    assert reply2.body is not None

    # Both invocations must have run through the factory.
    assert len(captured_streams) >= 2
    # Same id across calls — the core promise of persistent_stream.
    ids = {s.id for s in captured_streams}
    assert len(ids) == 1, f"persistent_stream must reuse the same stream id, got {ids}"
    # Same storage backend — the child can read prior history off the same store.
    storages = [s.history.storage for s in captured_streams]
    assert all(s is storages[0] for s in storages), "child streams must share storage"
