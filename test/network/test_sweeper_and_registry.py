# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Sweeper background firing + registry isolation + cross-tool flow.

* Real-clock expectation sweeper firing — separate from the unit
  tests that call ``hub._expectation_tick()`` directly; this exercise
  catches regressions in the background ``_IntervalSweeper`` event
  loop.
* TransitionRegistry isolation — custom targets registered on a fresh
  registry don't leak into the default singleton.
* Cross-tool flow exercising the LLM-tool surface end-to-end without
  a real LLM — verifies DI wiring across all 6 tools through one
  agent.
"""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import pytest

from ag2 import Agent, Context
from ag2.events import ToolCallEvent
from ag2.knowledge import MemoryKnowledgeStore
from ag2.network import (
    Hub,
    Resume,
)
from ag2.network.adapters.conversation import ConversationAdapter
from ag2.network.channel import (
    ChannelManifest,
    Expectation,
    ParticipantSchema,
)
from ag2.network.client.tools import (
    make_channels_tool,
    make_context_tool,
    make_peers_tool,
    make_tasks_tool,
)
from ag2.network.hub.audit import (
    AUDIT_KIND_EXPECTATION_VIOLATED,
)
from ag2.network.policies import AGENT_CLIENT_DEP
from ag2.network.transitions import (
    AgentTarget,
    TransitionDecision,
    TransitionGraph,
    TransitionRegistry,
    WorkflowGraphError,
)
from ag2.stream import MemoryStream
from ag2.testing import TestConfig


async def _invoke(tool: Any, args: dict, *, dependencies: dict | None = None) -> Any:
    """Invoke a ``FunctionTool`` directly and return the underlying value."""
    event = ToolCallEvent(name=tool.name, arguments=json.dumps(args))
    context = Context(stream=MemoryStream(), dependencies=dependencies or {})
    result_event = await tool(event, context)
    parts = getattr(result_event, "result", None)
    if parts is None or not parts.parts:
        return result_event
    first = parts.parts[0]
    if hasattr(first, "content"):
        return first.content
    if hasattr(first, "data"):
        return first.data
    return first


def _agent(name: str) -> Agent:
    return Agent(name=name, config=TestConfig())


# ── Real-clock expectation sweeper ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_expectation_sweeper_fires_violations_in_background() -> None:
    """The ``_IntervalSweeper`` background task evaluates expectations
    on a real interval and writes audit records — without anyone
    calling ``hub._expectation_tick()`` directly.

    This catches regressions in the sweeper loop (e.g. the loop being
    cancelled before its first ``fn`` call, or exceptions in
    ``_expectation_tick`` killing the background task) that mock-clock
    tests would miss.
    """
    immediate_manifest = ChannelManifest(
        type="conversation_immediate",
        version=1,
        participants=ParticipantSchema(min=2),
        expectations=[
            Expectation(
                name="max_silence",
                on_violation="audit",
                params={"seconds": 0},  # any silence triggers it
            ),
        ],
    )

    class _ImmediateAdapter(ConversationAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.manifest = immediate_manifest

    store = MemoryKnowledgeStore()
    # Sweeper runs every 50ms; default expectation evaluator is registered.
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0.05)
    hub.register_adapter(_ImmediateAdapter())

    alice = await hub.register(_agent("alice"))
    bob = await hub.register(_agent("bob"))

    pre_audit = len(await hub._audit_log.read_all())
    channel = await alice.open(type="conversation_immediate", target=bob.agent_id)

    # Real-clock wait — give the sweeper a few ticks to fire.
    await asyncio.sleep(0.25)

    audit = await hub._audit_log.read_all()
    new_records = audit[pre_audit:]
    violations = [
        r for r in new_records if r["kind"] == AUDIT_KIND_EXPECTATION_VIOLATED and r["channel_id"] == channel.channel_id
    ]
    assert len(violations) >= 1
    assert violations[0]["expectation"] == "max_silence"

    await hub.close()


# ── TransitionRegistry isolation ────────────────────────────────────────────


@dataclass(slots=True)
class _CustomIsoTarget:
    label: str
    name: ClassVar[str] = "custom_iso_target"

    def resolve(self, state, envelope):
        return TransitionDecision(next_speaker=None, close_reason=self.label)


def test_custom_registry_does_not_leak_into_default() -> None:
    """A custom :class:`TransitionRegistry` instance with custom types
    registered on it does not affect the default registry. The default
    registry's ``loads`` rejects unknown names; the custom one
    succeeds."""
    custom = TransitionRegistry()
    custom.register_target(_CustomIsoTarget)

    serialized = {
        "initial_speaker": "alice",
        "transitions": [
            {
                "when": {"name": "always", "args": {}},
                "then": {"name": "custom_iso_target", "args": {"label": "x"}},
                "priority": 0,
            }
        ],
        "default_target": {"name": "terminate", "args": {"reason": "fallback"}},
        "max_turns": None,
    }

    # Loaded with the custom registry → succeeds.
    graph = TransitionGraph.loads(serialized, registry=custom)
    assert graph.transitions[0].then == _CustomIsoTarget(label="x")

    # Loaded with a fresh default registry → fails (no custom target).
    fresh_default = TransitionRegistry()
    with pytest.raises(WorkflowGraphError, match="custom_iso_target"):
        TransitionGraph.loads(serialized, registry=fresh_default)


def test_default_transition_registry_singleton_is_lazy() -> None:
    """``TransitionRegistry.default()`` returns the same instance on
    repeat calls (singleton) and is constructed on first call rather
    than at import time."""
    r1 = TransitionRegistry.default()
    r2 = TransitionRegistry.default()
    assert r1 is r2
    # Built-ins resolve.
    target = r1.target_from_dict({"name": "agent", "args": {"agent_id": "alice"}})
    assert target == AgentTarget(agent_id="alice")


# ── Cross-tool flow (no LLM — direct FunctionTool invocation) ───────────────


@pytest.mark.asyncio
async def test_cross_tool_flow_exercises_all_six_tools() -> None:
    """One alice registers; she uses ``peers(find)``, ``channels(open)``,
    ``say``, ``tasks(list)``, ``context(search)``. The grouped+flat
    surface composes via the same DI wiring."""
    from ag2.network.client.tools.delegate import make_delegate_tool
    from ag2.network.client.tools.say import make_say_tool

    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(
        _agent("alice"),
        resume=Resume(claimed_capabilities=["lead"]),
    )
    bob = await hub.register(
        _agent("bob"),
        resume=Resume(claimed_capabilities=["math"]),
    )

    say = make_say_tool(alice)
    delegate = make_delegate_tool(alice)
    peers = make_peers_tool(alice)
    channels = make_channels_tool(alice)
    tasks = make_tasks_tool(alice)
    context = make_context_tool(alice)

    # All 6 tools created without raising and have the expected names.
    assert {t.name for t in (say, delegate, peers, channels, tasks, context)} == {
        "say",
        "delegate",
        "peers",
        "channels",
        "tasks",
        "context",
    }

    deps: dict = {AGENT_CLIENT_DEP: alice}

    # peers(action="find", capability="math") → bob
    found = await _invoke(peers, {"action": "find", "capability": "math"}, dependencies=deps)
    assert isinstance(found, list)
    assert any(p["name"] == "bob" for p in found)
    assert all(p["name"] != "alice" for p in found)  # excludes self

    # channels(action="open") → opens a conversation with bob
    opened = await _invoke(
        channels,
        {"action": "open", "type": "conversation", "target": bob.agent_id},
        dependencies=deps,
    )
    assert isinstance(opened, dict)
    channel_id = opened["channel_id"]

    # say(...) — post into the open channel.
    say_result = await _invoke(
        say,
        {"content": "hello bob", "channel_id": channel_id},
        dependencies=deps,
    )
    assert "posted envelope" in say_result

    # tasks(action="list", scope="own") — alice has no tasks.
    listed = await _invoke(tasks, {"action": "list", "scope": "own"}, dependencies=deps)
    assert listed == []

    # context(action="search", query="hello", scope="channel")
    found_msgs = await _invoke(
        context,
        {
            "action": "search",
            "query": "hello",
            "scope": "channel",
            "channel_id": channel_id,
        },
        dependencies=deps,
    )
    assert isinstance(found_msgs, list)
    assert len(found_msgs) == 1
    assert "hello" in found_msgs[0]["excerpt"]

    await hub.close()


# ── Trust-boundary check ────────────────────────────────────────────────────


def test_handlers_module_does_not_touch_hub_privates() -> None:
    """``handlers.py`` must not reach into ``_hub._<private>``.

    All hub access goes through :class:`HubClient`. This test fails if a
    future change re-introduces a hub-private shortcut — adapter / channel
    / task lookups, audit-log appends, listener fan-out, or any other
    underscore-prefixed attribute reach-through.
    """
    handlers_path = Path(__file__).parent.parent.parent / "ag2" / "network" / "client" / "handlers.py"
    text = handlers_path.read_text()
    forbidden = (
        "_hub._adapter_for",
        "_hub._adapter_states",
        "_hub._channels",
        "_hub._tasks",
        "_hub._fan_out",
        "_hub._audit_log",
        "_hub._listeners",
        "_hub._arbiter",
        "hub._fan_out",
        "hub._audit_log",
        "hub._listeners",
    )
    found = [pat for pat in forbidden if pat in text]
    assert not found, f"handlers.py must go through HubClient, not Hub privates: {found}"
