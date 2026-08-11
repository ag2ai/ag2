# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Grouped LLM tool tests (peers, channels, tasks, context).

Tools are tested by direct ``FunctionTool.__call__`` invocation with a
synthesised ``ToolCallEvent`` and a ``Context`` carrying the same
DI keys (``CHANNEL_DEP`` / ``AGENT_CLIENT_DEP``) that
``handlers.stamp_dependencies`` populates inside notify handlers. This
sidesteps the LLM and gives us deterministic per-action coverage.

A final integration test wires the full network plugin to two agents
and exercises the tools through real ``Agent.ask`` turns.
"""

import contextlib
import json
from typing import Any

import pytest

from ag2 import Agent, Context
from ag2.events import ToolCallEvent
from ag2.knowledge import MemoryKnowledgeStore
from ag2.network import (
    EV_TASK_CANCEL_REQUEST,
    Hub,
    Resume,
)
from ag2.network.client.tools.channels import make_channels_tool
from ag2.network.client.tools.context import make_context_tool
from ag2.network.client.tools.peers import make_peers_tool
from ag2.network.client.tools.tasks import make_tasks_tool
from ag2.network.policies import AGENT_CLIENT_DEP, CHANNEL_DEP
from ag2.stream import MemoryStream
from ag2.task import TaskMetadata, TaskSpec, TaskState
from ag2.testing import TestConfig


def _agent(name: str, *events: object) -> Agent:
    return Agent(name=name, config=TestConfig(*events))


async def _invoke(tool: Any, args: dict, *, dependencies: dict | None = None) -> Any:
    """Invoke a ``FunctionTool`` directly with ``args`` and return the underlying value.

    The framework wraps return values in ``ToolResult.parts`` — strings
    land as ``TextInput.content``, dict / list land as ``DataInput.data``.
    Tests want the raw value back, so this helper unwraps it.
    """
    event = ToolCallEvent(
        name=tool.name,
        arguments=json.dumps(args),
    )
    context = Context(stream=MemoryStream(), dependencies=dependencies or {})
    result_event = await tool(event, context)
    parts = getattr(result_event, "result", None)
    if parts is None or not parts.parts:
        # ToolErrorEvent path or empty result — surface the event for inspection.
        return result_event
    first = parts.parts[0]
    if hasattr(first, "content"):
        return first.content
    if hasattr(first, "data"):
        return first.data
    return first


# ── peers tool ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_peers_find_returns_other_peers_summary() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(_agent("alice"), resume=Resume(claimed_capabilities=["debate"]))
    await hub.register(
        _agent("bob"),
        resume=Resume(summary="senior coder", claimed_capabilities=["coding"]),
    )
    await hub.register(
        _agent("carol"),
        resume=Resume(summary="qa lead", claimed_capabilities=["testing"]),
    )

    tool = make_peers_tool(alice)
    deps = {AGENT_CLIENT_DEP: alice}
    result = await _invoke(tool, {"action": "find"}, dependencies=deps)

    names = [r["name"] for r in result]
    assert "bob" in names
    assert "carol" in names
    assert "alice" not in names  # excludes the calling agent

    await hub.close()


@pytest.mark.asyncio
async def test_peers_find_filters_by_capability() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(_agent("alice"))
    await hub.register(
        _agent("bob"),
        resume=Resume(claimed_capabilities=["coding"]),
    )

    tool = make_peers_tool(alice)
    deps = {AGENT_CLIENT_DEP: alice}
    coders = await _invoke(tool, {"action": "find", "capability": "coding"}, dependencies=deps)
    other = await _invoke(tool, {"action": "find", "capability": "missing"}, dependencies=deps)

    assert [r["name"] for r in coders] == ["bob"]
    assert other == []

    await hub.close()


@pytest.mark.asyncio
async def test_peers_describe_returns_skill_md_or_fallback() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(_agent("alice"))
    # Bob has an explicit SKILL.md.
    await hub.register(
        _agent("bob"),
        resume=Resume(claimed_capabilities=["coding"]),
        skill_md="---\nname: bob\ndescription: hand-written\n---\n## Notes\n",
    )
    # Carol falls back to the rendered version.
    await hub.register(
        _agent("carol"),
        resume=Resume(claimed_capabilities=["qa"], summary="qa lead"),
    )

    tool = make_peers_tool(alice)
    deps = {AGENT_CLIENT_DEP: alice}

    bob_profile = await _invoke(tool, {"action": "describe", "name": "bob"}, dependencies=deps)
    assert "hand-written" in bob_profile["skill_md"]
    assert bob_profile["resume"]["claimed_capabilities"] == ["coding"]

    carol_profile = await _invoke(tool, {"action": "describe", "name": "carol"}, dependencies=deps)
    assert "name: carol" in carol_profile["skill_md"]
    assert "qa lead" in carol_profile["skill_md"]

    await hub.close()


# ── channels tool ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_channels_open_and_list_and_close() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(_agent("alice"))
    bob = await hub.register(_agent("bob"))

    tool = make_channels_tool(alice)
    deps = {AGENT_CLIENT_DEP: alice}

    # Open via tool.
    opened = await _invoke(
        tool,
        {"action": "open", "type": "conversation", "target": "bob"},
        dependencies=deps,
    )
    sid = opened["channel_id"]

    # List shows it.
    listed = await _invoke(tool, {"action": "list"}, dependencies=deps)
    assert any(s["channel_id"] == sid for s in listed)

    # Info returns the full metadata.
    info = await _invoke(tool, {"action": "info", "channel_id": sid}, dependencies=deps)
    assert info["type"] == "conversation"
    assert info["state"] == "active"
    assert any(p["agent_id"] == bob.agent_id for p in info["participants"])

    # Close terminates.
    closed = await _invoke(tool, {"action": "close", "channel_id": sid}, dependencies=deps)
    assert closed["state"] == "closed"

    await hub.close()


# ── context tool ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_context_search_finds_substring_in_channel_wal() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(_agent("alice"))
    await hub.register(_agent("bob"))

    channel = await alice.open(type="conversation", target="bob")
    await channel.send("policy framework adoption")
    await channel.send("cost-benefit analysis")

    tool = make_context_tool(alice)
    deps = {AGENT_CLIENT_DEP: alice, CHANNEL_DEP: channel}
    results = await _invoke(tool, {"action": "search", "query": "framework"}, dependencies=deps)
    assert len(results) == 1
    assert "framework" in results[0]["excerpt"]

    await hub.close()


@pytest.mark.asyncio
async def test_context_quote_returns_recent_n_from_speaker() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(_agent("alice"), attach_plugin=False)
    bob = await hub.register(_agent("bob"), attach_plugin=False)

    # Auto-ack on bob so the conversation activates.
    from ag2.network import EV_CHANNEL_INVITE, EV_CHANNEL_INVITE_ACK, Envelope

    async def _ack(envelope: Envelope) -> None:
        if envelope.event_type != EV_CHANNEL_INVITE:
            return
        ack = Envelope(
            channel_id=envelope.channel_id,
            sender_id=bob.agent_id,
            audience=None,
            event_type=EV_CHANNEL_INVITE_ACK,
            event_data={"channel_id": envelope.channel_id},
            causation_id=envelope.envelope_id,
        )
        with contextlib.suppress(Exception):
            await bob.send_envelope(ack)

    bob.on_envelope(_ack)

    channel = await alice.open(type="conversation", target="bob")
    await channel.send("alice 1")
    await channel.send("alice 2")
    await channel.send("alice 3")

    tool = make_context_tool(alice)
    deps = {AGENT_CLIENT_DEP: alice, CHANNEL_DEP: channel}
    quotes = await _invoke(tool, {"action": "quote", "speaker": "alice", "recent_n": 2}, dependencies=deps)
    assert [q["text"] for q in quotes] == ["alice 2", "alice 3"]

    await hub.close()


# ── tasks tool ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tasks_status_and_list_and_wait() -> None:
    """Status / list / wait operate on hub-observed tasks."""
    from ag2.network.task_mirror import TaskMirror

    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    bob_agent = Agent(name="bob", config=TestConfig())
    bob = await hub.register(bob_agent)

    # Start + complete a task through the mirror so the hub sees it.
    stream = MemoryStream()
    mirror = TaskMirror(hub=hub, owner_id=bob.agent_id)
    sub_ids = mirror.attach(stream)
    try:
        async with bob_agent.task("indexing", context=Context(stream=stream)) as task:
            task_id = task.task_id
            await task.complete(result="ok")
    finally:
        mirror.detach(stream, sub_ids)

    tool = make_tasks_tool(bob)
    deps = {AGENT_CLIENT_DEP: bob}

    listed = await _invoke(tool, {"action": "list", "scope": "own", "state": "all"}, dependencies=deps)
    assert any(t["task_id"] == task_id for t in listed)

    status = await _invoke(tool, {"action": "status", "task_id": task_id}, dependencies=deps)
    assert status["state"] == "completed"
    assert status["result"] == "ok"

    waited = await _invoke(
        tool,
        {"action": "wait", "task_id": task_id, "timeout": 1.0, "poll_interval": 0.05},
        dependencies=deps,
    )
    assert waited["state"] == "completed"

    await hub.close()


@pytest.mark.asyncio
async def test_tasks_status_unknown_task_returns_error() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(_agent("alice"))

    tool = make_tasks_tool(alice)
    deps = {AGENT_CLIENT_DEP: alice}
    result = await _invoke(tool, {"action": "status", "task_id": "nonexistent"}, dependencies=deps)
    assert isinstance(result, str)
    assert "not found" in result

    await hub.close()


@pytest.mark.asyncio
async def test_tasks_cancel_posts_cancel_request_envelope_to_owner() -> None:
    """``tasks(action="cancel")`` posts an ``ag2.task.cancel_request``
    envelope into the task's channel addressed to the owner. The owner
    is free to honour or ignore — the framework only delivers the ask.
    """
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(_agent("alice"))
    bob = await hub.register(_agent("bob"))

    channel = await alice.open(type="conversation", target="bob")

    # Plant a long-running task in the hub on bob's behalf, associated with
    # the open channel so the cancel verb has somewhere to ride.
    await hub.observe_task(
        TaskMetadata(
            task_id="task-bob-1",
            owner_id=bob.agent_id,
            spec=TaskSpec(title="long index"),
            state=TaskState.RUNNING,
            channel_id=channel.channel_id,
        )
    )

    tool = make_tasks_tool(alice)
    deps = {AGENT_CLIENT_DEP: alice}
    result = await _invoke(
        tool,
        {"action": "cancel", "task_id": "task-bob-1", "reason": "wrap up"},
        dependencies=deps,
    )
    assert isinstance(result, str)
    assert "cancel_request posted" in result

    wal = await hub.read_wal(channel.channel_id)
    requests = [e for e in wal if e.event_type == EV_TASK_CANCEL_REQUEST]
    assert len(requests) == 1
    assert requests[0].event_data == {"task_id": "task-bob-1", "reason": "wrap up"}
    assert requests[0].audience == [bob.agent_id]
    assert requests[0].sender_id == alice.agent_id

    await hub.close()


@pytest.mark.asyncio
async def test_tasks_cancel_without_task_id_returns_error() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(_agent("alice"))

    tool = make_tasks_tool(alice)
    deps = {AGENT_CLIENT_DEP: alice}
    result = await _invoke(tool, {"action": "cancel"}, dependencies=deps)
    assert isinstance(result, str)
    assert "requires `task_id`" in result

    await hub.close()


@pytest.mark.asyncio
async def test_tasks_cancel_unknown_task_returns_error() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(_agent("alice"))

    tool = make_tasks_tool(alice)
    deps = {AGENT_CLIENT_DEP: alice}
    result = await _invoke(tool, {"action": "cancel", "task_id": "ghost"}, dependencies=deps)
    assert isinstance(result, str)
    assert "not found" in result

    await hub.close()


@pytest.mark.asyncio
async def test_tasks_cancel_terminal_task_is_a_no_op_with_message() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(_agent("alice"))
    bob = await hub.register(_agent("bob"))

    # Plant a COMPLETED task; cancel must report it without posting an envelope.
    await hub.observe_task(
        TaskMetadata(
            task_id="task-done",
            owner_id=bob.agent_id,
            spec=TaskSpec(title="x"),
            state=TaskState.COMPLETED,
        )
    )

    tool = make_tasks_tool(alice)
    deps = {AGENT_CLIENT_DEP: alice}
    result = await _invoke(tool, {"action": "cancel", "task_id": "task-done"}, dependencies=deps)
    assert isinstance(result, str)
    assert "already completed" in result

    await hub.close()


@pytest.mark.asyncio
async def test_tasks_cancel_task_without_channel_returns_error() -> None:
    """Cancel-requests ride on the task's channel; without one there's
    nowhere to address the envelope."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(_agent("alice"))
    bob = await hub.register(_agent("bob"))

    await hub.observe_task(
        TaskMetadata(
            task_id="task-orphan",
            owner_id=bob.agent_id,
            spec=TaskSpec(title="standalone"),
            state=TaskState.RUNNING,
        )
    )

    tool = make_tasks_tool(alice)
    deps = {AGENT_CLIENT_DEP: alice}
    result = await _invoke(tool, {"action": "cancel", "task_id": "task-orphan"}, dependencies=deps)
    assert isinstance(result, str)
    assert "no associated channel" in result

    await hub.close()


# ── Plugin attaches all 6 tools ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_network_plugin_attaches_identity_level_tools() -> None:
    """``NetworkPlugin`` attaches the identity-level cross-cutting tools only.

    ``say`` is channel-shaped: it comes from
    ``adapter.tools_for(...)`` per turn (the default notify handler
    resolves and merges it into ``agent.ask(tools=...)``). Workflow
    agents see ``[]`` from the adapter — they never see ``say``.
    """
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(_agent("alice"))
    tool_names = {t.name for t in alice.agent.tools}
    assert {"delegate", "peers", "channels", "tasks", "context"} <= tool_names
    assert "say" not in tool_names

    await hub.close()
