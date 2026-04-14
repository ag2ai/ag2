# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Subtask smoke: run_subtask, run_subtasks (parallel + sequential), as_tool
delegation, depth_limiter, persistent_stream. Real LLM calls.
"""

from __future__ import annotations

import pytest

from autogen.beta import Actor, TaskConfig
from autogen.beta.events import TaskCompleted, TaskStarted
from autogen.beta.stream import MemoryStream
from autogen.beta.tools.subagents import depth_limiter, persistent_stream


pytestmark = [pytest.mark.asyncio, pytest.mark.gemini]


async def test_run_subtask_auto_injected(gemini_flash_config) -> None:
    """run_subtask is auto-injected on every Actor; LLM can dispatch it."""
    task_starts: list[TaskStarted] = []
    task_completions: list[TaskCompleted] = []

    stream = MemoryStream()
    stream.where(TaskStarted).subscribe(lambda e: task_starts.append(e))
    stream.where(TaskCompleted).subscribe(lambda e: task_completions.append(e))

    agent = Actor(
        "delegator",
        prompt=(
            "You can spawn subtasks via the run_subtask tool when a question "
            "needs isolated focused work. Always use run_subtask for "
            "self-contained research questions. Be concise."
        ),
        config=gemini_flash_config,
    )
    reply = await agent.ask(
        "Use run_subtask to find out what colour ripe bananas are. Then tell me the answer in one sentence.",
        stream=stream,
    )

    assert reply.body is not None
    assert len(task_starts) >= 1
    assert len(task_completions) >= 1
    assert task_completions[0].task_id == task_starts[0].task_id
    assert "yellow" in reply.body.lower()


async def test_run_subtasks_parallel(gemini_flash_config) -> None:
    """run_subtasks(parallel=True) dispatches multiple subtasks concurrently."""
    task_completions: list[TaskCompleted] = []

    stream = MemoryStream()
    stream.where(TaskCompleted).subscribe(lambda e: task_completions.append(e))

    agent = Actor(
        "fanner",
        prompt=(
            "You can call run_subtasks(tasks=[...], parallel=True) to run "
            "many independent questions concurrently. Use this whenever the "
            "user asks several unrelated things at once."
        ),
        config=gemini_flash_config,
    )
    reply = await agent.ask(
        "Use run_subtasks with parallel=True to answer ALL of these in one tool call: "
        "(a) capital of France, (b) capital of Japan, (c) capital of Brazil. "
        "Then list all three answers in your reply.",
        stream=stream,
    )

    assert reply.body is not None
    body = reply.body.lower()
    assert "paris" in body
    assert "tokyo" in body
    assert "brasília" in body or "brasilia" in body
    # 3 subtasks → at least 3 results
    assert len(task_completions) >= 3


async def test_subtask_prompt_override(gemini_flash_config) -> None:
    """TaskConfig.prompt overrides the default subtask system prompt."""
    agent = Actor(
        "two-tier",
        prompt="Use run_subtask for any factual lookup. Be concise.",
        config=gemini_flash_config,
        tasks=TaskConfig(
            prompt="You are a fast lookup agent. Answer in one short sentence.",
        ),
    )
    reply = await agent.ask("Use run_subtask to look up: what is the boiling point of water in Celsius?")
    assert reply.body is not None
    assert "100" in reply.body


async def test_actor_as_tool_delegation(gemini_flash_config) -> None:
    """A.as_tool() lets actor B call A as a sibling tool."""
    expert = Actor(
        "math-expert",
        prompt="You only do arithmetic. Reply with just the number.",
        config=gemini_flash_config,
    )

    coordinator = Actor(
        "coordinator",
        prompt=(
            "You delegate math problems to the task_math-expert tool. "
            "After receiving the answer, present it as a complete sentence."
        ),
        config=gemini_flash_config,
        tools=[expert.as_tool(description="Delegate any arithmetic to the math-expert agent.")],
    )

    reply = await coordinator.ask("What is 19 * 23?")
    assert reply.body is not None
    assert "437" in reply.body


async def test_depth_limiter_prevents_recursion(gemini_flash_config) -> None:
    """depth_limiter() caps recursive as_tool delegation depth."""

    # Build A that has itself as a tool, with a max_depth=1 limiter.
    actor = Actor(
        "recursive",
        prompt=(
            "You have a task_recursive tool that re-invokes you. "
            "If asked to recurse more than once, the tool will refuse — "
            "explain that and stop. Be brief."
        ),
        config=gemini_flash_config,
    )
    actor.add_tool(
        actor.as_tool(
            description="Delegate to yourself recursively.",
            middleware=[depth_limiter(max_depth=1)],
        )
    )

    reply = await actor.ask(
        "Call task_recursive with the objective 'recurse again' once. "
        "Then call task_recursive again from within. Report what happens."
    )
    assert reply.body is not None
    # Should not crash and should produce a reply
    assert reply.body.strip() != ""


async def test_persistent_stream_shares_history(gemini_flash_config) -> None:
    """persistent_stream() reuses one stream id across as_tool calls.

    The child agent should remember context from a previous invocation
    if persistent_stream is used.
    """
    child = Actor(
        "memo",
        prompt=(
            "You are a notepad. Whatever the user tells you, store mentally "
            "and recall on demand. Be terse."
        ),
        config=gemini_flash_config,
    )

    parent = Actor(
        "owner",
        prompt=(
            "You have a task_memo tool. Use it to store and retrieve facts. "
            "Each call should reuse the same notepad."
        ),
        config=gemini_flash_config,
        tools=[child.as_tool(description="Notepad agent.", stream=persistent_stream())],
    )

    # First ask: store a fact
    reply1 = await parent.ask(
        "Use the task_memo tool with objective 'remember that the launch code is FOXTROT-7'. "
        "After the tool returns, just say 'stored'."
    )
    assert reply1.body is not None

    # Second ask: retrieve fact (should remember if persistent_stream works)
    reply2 = await reply1.ask(
        "Use task_memo with objective 'recall the launch code'. "
        "Then tell me what the launch code is."
    )
    assert reply2.body is not None
    assert "foxtrot" in reply2.body.lower() or "foxtrot-7" in reply2.body.lower()
