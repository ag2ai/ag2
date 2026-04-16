# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Phase 6 — task verb unit tests (run_task, track_task)."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest
import pytest_asyncio

from autogen.beta.context import ConversationContext
from autogen.beta.events import ToolCallEvent, ToolResultEvent
from autogen.beta.knowledge import MemoryKnowledgeStore
from autogen.beta.network import (
    SESSION_DEP,
    AccessBlock,
    ActorIdentity,
    HubClient,
    LocalLink,
    Rule,
    SessionType,
    Task,
    TaskState,
)
from autogen.beta.network.client.session import Session
from autogen.beta.network.client.verbs.task import (
    _summarise_task,
    build_task_verbs,
)
from autogen.beta.network.hub import Hub


@dataclass
class _FakeReply:
    content: str


class _FakeActor:
    def __init__(self, name: str, reply: str = "") -> None:
        self.name = name
        self.reply = reply

    async def ask(self, content: str, **kwargs: Any) -> _FakeReply:
        return _FakeReply(content=self.reply or f"{self.name}: {content}")


@pytest_asyncio.fixture
async def task_setup():
    """Two clients on one hub plus an active alice→bob consulting session.

    Bob has a custom ``research`` task handler that completes
    immediately so ``run_task(blocking=True)`` works deterministically.
    """

    store = MemoryKnowledgeStore()
    hub = Hub(store)
    link = LocalLink()
    link.on_connection(hub.connection_handler)
    hub_client = HubClient(hub, link)

    alice_client = await hub_client.register(
        _FakeActor("alice"),
        identity=ActorIdentity(name="alice", capabilities=["ask"]),
        rule=Rule(access=AccessBlock(outbound_to=["*"])),
    )
    bob_client = await hub_client.register(
        _FakeActor("bob"),
        identity=ActorIdentity(name="bob", capabilities=["answer"]),
        rule=Rule(access=AccessBlock(outbound_to=["*"])),
    )

    @bob_client.on_task("research")
    async def _research_handler(envelope, task: Task, client) -> None:  # noqa: ANN001
        await task.phase_entered("plan")
        await task.result(f"answered: {task.metadata.spec.description}")

    @bob_client.on_task("loop")
    async def _slow_handler(envelope, task: Task, client) -> None:  # noqa: ANN001
        await task.phase_entered("forever")
        await asyncio.sleep(60)

    @bob_client.on_task("oops")
    async def _crash_handler(envelope, task: Task, client) -> None:  # noqa: ANN001
        raise RuntimeError("intentional handler crash")

    session = await alice_client.open(SessionType.CONSULTING, target="bob")

    try:
        yield hub, alice_client, bob_client, session
    finally:
        await hub_client.close()
        await link.close()


def _ctx(session: Session) -> ConversationContext:
    return ConversationContext(
        stream=MagicMock(),
        dependencies={SESSION_DEP: session},
    )


async def _call(tool: Any, ctx: ConversationContext, **arguments: Any) -> Any:
    event = ToolCallEvent(
        name=tool.schema.function.name,
        arguments=json.dumps(arguments),
    )
    result: ToolResultEvent = await tool(event, ctx)
    return result.result.content


def _verbs(client: Any) -> dict[str, Any]:
    return {v.schema.function.name: v for v in build_task_verbs(client)}


@pytest.mark.asyncio
class TestRunTask:
    async def test_blocking_returns_completed_result(
        self, task_setup
    ) -> None:
        _hub, alice_client, _bob, session = task_setup
        verbs = _verbs(alice_client)
        ctx = _ctx(session)

        result = await _call(
            verbs["run_task"],
            ctx,
            title="research a topic",
            description="What are the latest CRISPR results?",
            spec_type="research",
        )
        assert result["state"] == "completed"
        assert "answered" in str(result["result"])
        assert "task_id" in result

    async def test_non_blocking_returns_handle_immediately(
        self, task_setup
    ) -> None:
        _hub, alice_client, _bob, session = task_setup
        verbs = _verbs(alice_client)
        ctx = _ctx(session)

        result = await _call(
            verbs["run_task"],
            ctx,
            title="t",
            description="d",
            spec_type="research",
            blocking=False,
        )
        assert "task_id" in result
        # Could be created/running/completed depending on scheduler — all
        # are valid non-blocking states.
        assert result["state"] in {"created", "running", "completed"}

    async def test_blocking_handler_crash_returns_failed(
        self, task_setup
    ) -> None:
        _hub, alice_client, _bob, session = task_setup
        verbs = _verbs(alice_client)
        ctx = _ctx(session)

        result = await _call(
            verbs["run_task"],
            ctx,
            title="t",
            description="d",
            spec_type="oops",
        )
        assert result["state"] == "failed"
        assert "error" in result
        assert "RuntimeError" in result["error"]

    async def test_blocking_with_timeout_returns_timeout_error(
        self, task_setup
    ) -> None:
        _hub, alice_client, _bob, session = task_setup
        verbs = _verbs(alice_client)
        ctx = _ctx(session)

        result = await _call(
            verbs["run_task"],
            ctx,
            title="t",
            description="d",
            spec_type="loop",
            blocking=True,
            timeout=0.2,
        )
        assert "error" in result
        assert "timeout" in result["error"]

    async def test_unknown_session_returns_error(self, task_setup) -> None:
        _hub, alice_client, _bob, _session = task_setup
        verbs = _verbs(alice_client)
        ctx = ConversationContext(stream=MagicMock())  # no SESSION_DEP

        result = await _call(
            verbs["run_task"],
            ctx,
            title="t",
            description="d",
        )
        assert "error" in result

    async def test_default_spec_type_routes_to_default_handler(
        self, task_setup
    ) -> None:
        _hub, alice_client, _bob, session = task_setup
        verbs = _verbs(alice_client)
        ctx = _ctx(session)

        # Empty spec_type → "*" fallback → default handler that calls
        # actor.ask. _FakeActor returns "bob: <description>".
        result = await _call(
            verbs["run_task"],
            ctx,
            title="t",
            description="hello",
        )
        assert result["state"] == "completed"
        assert "bob: hello" in str(result["result"])


@pytest.mark.asyncio
class TestTrackTask:
    async def test_track_known_task(self, task_setup) -> None:
        _hub, alice_client, _bob, session = task_setup
        verbs = _verbs(alice_client)
        ctx = _ctx(session)

        run = await _call(
            verbs["run_task"],
            ctx,
            title="t",
            description="d",
            spec_type="research",
        )
        looked_up = await _call(verbs["track_task"], ctx, task_id=run["task_id"])
        assert looked_up["task_id"] == run["task_id"]
        assert looked_up["state"] == "completed"

    async def test_track_unknown_returns_error(self, task_setup) -> None:
        _hub, alice_client, _bob, session = task_setup
        verbs = _verbs(alice_client)
        ctx = _ctx(session)

        result = await _call(
            verbs["track_task"], ctx, task_id="01000000-0000-0000-0000-000000000000"
        )
        assert "error" in result


class TestSummariseTask:
    """Pin the task summary keys — the LLM relies on them."""

    def test_summary_keys_are_stable(self, task_setup=None) -> None:
        from autogen.beta.network.task import TaskMetadata, TaskSpec

        spec = TaskSpec(title="t", description="d")
        meta = TaskMetadata(
            task_id="task-1",
            session_id="sess-1",
            owner_id="bob",
            requester_id="alice",
            spec=spec,
            state=TaskState.COMPLETED,
            current_phase="done",
            progress={"pct": 1.0},
            result="payload",
            error=None,
            created_at="2026-04-14T00:00:00Z",
            started_at="2026-04-14T00:00:01Z",
            completed_at="2026-04-14T00:00:02Z",
            expires_at="2026-04-14T01:00:00Z",
        )
        summary = _summarise_task(meta)
        assert summary == {
            "task_id": "task-1",
            "session_id": "sess-1",
            "owner_id": "bob",
            "requester_id": "alice",
            "state": "completed",
            "current_phase": "done",
            "progress": {"pct": 1.0},
            "result": "payload",
            "error": None,
            "created_at": "2026-04-14T00:00:00Z",
            "started_at": "2026-04-14T00:00:01Z",
            "completed_at": "2026-04-14T00:00:02Z",
            "expires_at": "2026-04-14T01:00:00Z",
        }
