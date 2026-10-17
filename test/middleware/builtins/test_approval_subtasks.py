# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import pytest

from ag2 import Agent, TaskConfig
from ag2.events import HumanInputRequest, HumanMessage, ToolCallEvent
from ag2.middleware import approval_required
from ag2.testing import TestConfig
from ag2.tools import tool
from ag2.tools.final import FunctionTool

TOOL_NAMES = ("deploy", "read_logs", "purge_cache")


def _human(answers: dict[str, str]) -> "tuple[Callable[[HumanInputRequest], HumanMessage], list[str]]":
    """A ``hitl_hook`` answering per tool name, and the tools it was asked about, in order."""
    asked: list[str] = []

    def hook(event: HumanInputRequest) -> HumanMessage:
        name = next(n for n in TOOL_NAMES if f"`{n}`" in event.content)
        asked.append(name)
        return HumanMessage(answers[name])

    return hook, asked


def _gated_tools() -> "tuple[dict[str, FunctionTool], list[str]]":
    """Approval-gated tools by name, and the names of the ones that actually ran, in order."""
    executed: list[str] = []

    def deploy() -> str:
        executed.append("deploy")
        return "deployed"

    def read_logs() -> str:
        executed.append("read_logs")
        return "logs"

    def purge_cache() -> str:
        executed.append("purge_cache")
        return "purged"

    tools = {fn.__name__: tool(fn, middleware=[approval_required()]) for fn in (deploy, read_logs, purge_cache)}
    return tools, executed


def _delegate(sub: Agent) -> ToolCallEvent:
    return ToolCallEvent(name=f"task_{sub.name}", arguments='{"objective": "do it"}')


@pytest.mark.asyncio()
async def test_parent_always_does_not_approve_the_subagents_call() -> None:
    tools, executed = _gated_tools()
    alice, alice_asked = _human({"deploy": "always"})
    bob, bob_asked = _human({"deploy": "n"})
    sub = Agent("sub", config=TestConfig(ToolCallEvent(name="deploy"), "done"), tools=[tools["deploy"]], hitl_hook=bob)
    parent = Agent(
        "parent",
        config=TestConfig(ToolCallEvent(name="deploy"), _delegate(sub), "done"),
        tools=[tools["deploy"], sub.as_tool(description="sub")],
        hitl_hook=alice,
    )

    await parent.ask("Hi!")

    assert alice_asked == ["deploy"]
    assert bob_asked == ["deploy"]
    assert executed == ["deploy"]


@pytest.mark.asyncio()
async def test_subagent_always_does_not_approve_the_parents_call() -> None:
    tools, executed = _gated_tools()
    alice, alice_asked = _human({"read_logs": "always", "purge_cache": "n"})
    bob, bob_asked = _human({"purge_cache": "always"})
    sub = Agent(
        "sub",
        config=TestConfig(ToolCallEvent(name="purge_cache"), "done"),
        tools=[tools["purge_cache"]],
        hitl_hook=bob,
    )
    parent = Agent(
        "parent",
        config=TestConfig(
            ToolCallEvent(name="read_logs"),
            _delegate(sub),
            ToolCallEvent(name="purge_cache"),
            "done",
        ),
        tools=[tools["read_logs"], tools["purge_cache"], sub.as_tool(description="sub")],
        hitl_hook=alice,
    )

    await parent.ask("Hi!")

    assert bob_asked == ["purge_cache"]
    assert alice_asked == ["read_logs", "purge_cache"]
    assert executed == ["read_logs", "purge_cache"]


@pytest.mark.asyncio()
async def test_sibling_always_does_not_approve_the_next_siblings_call() -> None:
    tools, executed = _gated_tools()
    human, asked = _human({"read_logs": "always", "purge_cache": "always"})
    parent = Agent(
        "parent",
        config=TestConfig(
            ToolCallEvent(name="read_logs"),
            ToolCallEvent(name="run_subtasks", arguments='{"tasks": ["a", "b"], "parallel": false}'),
            "done",
        ),
        tools=[tools["read_logs"], tools["purge_cache"]],
        tasks=TaskConfig(config=TestConfig(ToolCallEvent(name="purge_cache"), "done")),
        hitl_hook=human,
    )

    await parent.ask("Hi!")

    assert asked == ["read_logs", "purge_cache", "purge_cache"]
    assert executed == ["read_logs", "purge_cache", "purge_cache"]


@pytest.mark.asyncio()
async def test_subagent_always_does_not_persist_into_the_parents_next_turn() -> None:
    tools, executed = _gated_tools()
    human, asked = _human({"read_logs": "always", "purge_cache": "always"})
    sub = Agent(
        "sub",
        config=TestConfig(ToolCallEvent(name="purge_cache"), "done"),
        tools=[tools["purge_cache"]],
    )
    parent = Agent(
        "parent",
        config=TestConfig(
            ToolCallEvent(name="read_logs"),
            _delegate(sub),
            "done",
            ToolCallEvent(name="purge_cache"),
            "done again",
        ),
        tools=[tools["read_logs"], tools["purge_cache"], sub.as_tool(description="sub")],
        hitl_hook=human,
    )

    reply = await parent.ask("Hi!")
    await reply.ask("again")

    assert asked == ["read_logs", "purge_cache", "purge_cache"]
    assert executed == ["read_logs", "purge_cache", "purge_cache"]


@pytest.mark.asyncio()
async def test_subagent_always_stays_in_the_subagent_without_a_parent_always() -> None:
    tools, executed = _gated_tools()
    human, asked = _human({"purge_cache": "always"})
    sub = Agent(
        "sub",
        config=TestConfig(ToolCallEvent(name="purge_cache"), ToolCallEvent(name="purge_cache"), "done"),
        tools=[tools["purge_cache"]],
    )
    parent = Agent(
        "parent",
        config=TestConfig(_delegate(sub), ToolCallEvent(name="purge_cache"), "done"),
        tools=[tools["purge_cache"], sub.as_tool(description="sub")],
        hitl_hook=human,
    )

    await parent.ask("Hi!")

    assert asked == ["purge_cache", "purge_cache"]
    assert executed == ["purge_cache", "purge_cache", "purge_cache"]
