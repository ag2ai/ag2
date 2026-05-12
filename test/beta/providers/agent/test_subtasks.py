# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Subtask smoke: run_subtask, run_subtasks (parallel + sequential), as_tool
delegation, no-recursion guarantee, persistent_stream. Real LLM calls.
"""

import pytest

from autogen.beta import Agent
from autogen.beta import agent as actor_mod
from autogen.beta.agent import TaskConfig
from autogen.beta.events import TaskCompleted, TaskStarted
from autogen.beta.history import MemoryStorage
from autogen.beta.stream import MemoryStream
from autogen.beta.tools.subagents import persistent_stream
from autogen.beta.tools.subagents import run_task as run_task_mod

pytestmark = pytest.mark.asyncio


async def test_run_subtask_auto_injected(provider_config) -> None:
    """``tasks=TaskConfig()`` injects ``run_subtask``; LLM can dispatch it."""
    task_starts: list[TaskStarted] = []
    task_completions: list[TaskCompleted] = []

    stream = MemoryStream()
    stream.where(TaskStarted).subscribe(lambda e: task_starts.append(e))
    stream.where(TaskCompleted).subscribe(lambda e: task_completions.append(e))

    agent = Agent(
        "delegator",
        prompt=(
            "You can spawn subtasks via the run_subtask tool when a question "
            "needs isolated focused work. Always use run_subtask for "
            "self-contained research questions. Be concise."
        ),
        config=provider_config,
        tasks=TaskConfig(),
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


async def test_run_subtasks_parallel(provider_config) -> None:
    """run_subtasks(parallel=True) dispatches multiple subtasks concurrently."""
    task_completions: list[TaskCompleted] = []

    stream = MemoryStream()
    stream.where(TaskCompleted).subscribe(lambda e: task_completions.append(e))

    agent = Agent(
        "fanner",
        prompt=(
            "You can call run_subtasks(tasks=[...], parallel=True) to run "
            "many independent questions concurrently. Use this whenever the "
            "user asks several unrelated things at once."
        ),
        config=provider_config,
        tasks=TaskConfig(),
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


async def test_subtask_prompt_override(provider_config) -> None:
    """TaskConfig.prompt overrides the default subtask system prompt.

    Verified by injecting a unique watermark token into the override prompt
    and asserting the subtask's TaskCompleted.result contains it. Looking at
    the parent's ``reply.body`` would be insufficient — the parent agent may
    strip the marker when summarising.
    """
    completions: list[TaskCompleted] = []
    stream = MemoryStream()
    stream.where(TaskCompleted).subscribe(lambda e: completions.append(e))

    agent = Agent(
        "two-tier",
        prompt="Use run_subtask for any factual lookup. Be concise.",
        config=provider_config,
        tasks=TaskConfig(
            prompt=(
                "You are a fast lookup agent. ALWAYS begin every reply with the "
                "literal token [WATERMARK_42]. Then answer in one short sentence."
            ),
        ),
    )
    reply = await agent.ask(
        "Use run_subtask to look up: what is the boiling point of water in Celsius?",
        stream=stream,
    )
    assert reply.body is not None
    assert completions, "subtask must have completed"
    # The override prompt forces the subtask to emit the watermark verbatim —
    # without the override, the default subtask prompt would not include it.
    assert any("[WATERMARK_42]" in (c.result or "") for c in completions)
    # And the actual answer is still produced
    assert any("100" in (c.result or "") for c in completions)


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


async def test_subtask_cannot_recurse(provider_config) -> None:
    """Subtasks structurally lack ``run_subtask`` — no recursion possible.

    The parent has ``run_subtask``; we instruct it to delegate. Inside the
    spawned subtask Agent we inspect the tool surface: ``run_subtask`` /
    ``run_subtasks`` must be absent, and ``_task_config`` must be ``None``.
    """
    captured_subtasks: list[Agent] = []

    original = run_task_mod.run_task

    async def capturing_run_task(agent, *args, **kwargs):
        captured_subtasks.append(agent)
        return await original(agent, *args, **kwargs)

    run_task_mod.run_task = capturing_run_task
    actor_mod._run_task = capturing_run_task
    try:
        agent = Agent(
            "delegator",
            prompt="Always use run_subtask for any factual lookup. Be concise.",
            config=provider_config,
            tasks=TaskConfig(),
        )
        await agent.ask("Use run_subtask to look up: what colour is the sky?")
    finally:
        run_task_mod.run_task = original
        actor_mod._run_task = original

    assert captured_subtasks, "subtask must have been spawned"
    child = captured_subtasks[0]
    assert child._task_config is None
    assert child._build_subtask_tools() == []


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
