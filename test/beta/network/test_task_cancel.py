# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Task cancellation — Phase 4.

Cancellation is a direct :meth:`Hub.cancel_task` call (not a wire envelope)
that can come from either the requester or the owner. The hub emits an
``ag2.task.cancelled`` broadcast so subscribers and session participants
see the transition.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest
import pytest_asyncio

from autogen.beta.knowledge import MemoryKnowledgeStore
from autogen.beta.network import (
    AccessDeniedError,
    ActorIdentity,
    Hub,
    HubClient,
    LocalLink,
    SessionType,
    Task,
    TaskSpec,
    TaskState,
    UnknownTaskError,
)


@dataclass
class FakeReply:
    content: str


class FakeActor:
    def __init__(self, name: str) -> None:
        self.name = name

    async def ask(self, content: str, **kwargs: Any) -> FakeReply:
        return FakeReply(content=f"{self.name}: {content}")


@pytest_asyncio.fixture
async def wired():
    store = MemoryKnowledgeStore()
    hub = Hub(store)
    link = LocalLink()
    link.on_connection(hub.connection_handler)
    hub_client = HubClient(hub, link)

    alice = FakeActor(name="alice")
    bob = FakeActor(name="bob")
    carol = FakeActor(name="carol")
    alice_client = await hub_client.register(
        alice, identity=ActorIdentity(name="alice")
    )
    bob_client = await hub_client.register(
        bob, identity=ActorIdentity(name="bob")
    )
    carol_client = await hub_client.register(
        carol, identity=ActorIdentity(name="carol")
    )
    try:
        yield hub, hub_client, alice_client, bob_client, carol_client
    finally:
        await hub_client.close()
        await link.close()


class TestCancel:
    @pytest.mark.asyncio
    async def test_requester_can_cancel(self, wired) -> None:
        hub, _hc, alice_client, bob_client, _carol_client = wired
        release = asyncio.Event()

        @bob_client.on_task("hold")
        async def _run(envelope, task: Task, client) -> None:  # noqa: ANN001
            await task.phase_entered("wait")
            await release.wait()

        sess = await alice_client.open(SessionType.CONSULTING, target="bob")
        task = await sess.create_task(
            TaskSpec(title="hold", spec_type="hold"),
            owner="bob",
            blocking=False,
        )
        for _ in range(50):
            if hub.peek_task(task.task_id).state is TaskState.RUNNING:
                break
            await asyncio.sleep(0.01)

        updated = await task.cancel(reason="no longer needed")
        assert updated.state is TaskState.CANCELLED
        assert hub.peek_task(task.task_id).state is TaskState.CANCELLED
        release.set()

    @pytest.mark.asyncio
    async def test_owner_can_cancel(self, wired) -> None:
        hub, _hc, alice_client, bob_client, _carol_client = wired
        release = asyncio.Event()
        inside_handler: dict[str, Task] = {}

        @bob_client.on_task("self-cancel")
        async def _run(envelope, task: Task, client) -> None:  # noqa: ANN001
            inside_handler["task"] = task
            await task.phase_entered("wait")
            await release.wait()

        sess = await alice_client.open(SessionType.CONSULTING, target="bob")
        task = await sess.create_task(
            TaskSpec(title="t", spec_type="self-cancel"),
            owner="bob",
            blocking=False,
        )
        for _ in range(50):
            if "task" in inside_handler:
                break
            await asyncio.sleep(0.01)

        # Owner cancels from inside the handler.
        updated = await inside_handler["task"].cancel(reason="abort")
        assert updated.state is TaskState.CANCELLED
        release.set()

    @pytest.mark.asyncio
    async def test_non_requester_non_owner_cannot_cancel(self, wired) -> None:
        hub, hub_client, alice_client, bob_client, carol_client = wired
        release = asyncio.Event()

        @bob_client.on_task("hold")
        async def _run(envelope, task: Task, client) -> None:  # noqa: ANN001
            await release.wait()

        sess = await alice_client.open(SessionType.CONSULTING, target="bob")
        task = await sess.create_task(
            TaskSpec(title="t", spec_type="hold"),
            owner="bob",
            blocking=False,
        )
        # Carol is neither requester nor owner.
        with pytest.raises(AccessDeniedError):
            await hub.cancel_task(
                task.task_id, requested_by=carol_client.actor_id, reason="not me"
            )
        release.set()

    @pytest.mark.asyncio
    async def test_cancel_completed_task_is_noop(self, wired) -> None:
        hub, _hc, alice_client, bob_client, _carol_client = wired

        @bob_client.on_task("fast")
        async def _run(envelope, task: Task, client) -> None:  # noqa: ANN001
            await task.result("done")

        sess = await alice_client.open(SessionType.CONSULTING, target="bob")
        terminal = await sess.create_task(
            TaskSpec(title="t", spec_type="fast"),
            owner="bob",
            blocking=True,
            timeout=2.0,
        )
        assert terminal.state is TaskState.COMPLETED

        again = await hub.cancel_task(
            terminal.task_id, requested_by=alice_client.actor_id, reason="late"
        )
        assert again.state is TaskState.COMPLETED  # unchanged — terminal idempotent

    @pytest.mark.asyncio
    async def test_cancel_unknown_task_raises(self, wired) -> None:
        hub, _hc, alice_client, _bob_client, _carol_client = wired
        with pytest.raises(UnknownTaskError):
            await hub.cancel_task(
                "01932nope", requested_by=alice_client.actor_id, reason=""
            )

    @pytest.mark.asyncio
    async def test_cancel_fans_out_to_session_subscribers(self, wired) -> None:
        hub, _hc, alice_client, bob_client, _carol_client = wired
        release = asyncio.Event()

        @bob_client.on_task("hold")
        async def _run(envelope, task: Task, client) -> None:  # noqa: ANN001
            await release.wait()

        sess = await alice_client.open(SessionType.CONSULTING, target="bob")
        task = await sess.create_task(
            TaskSpec(title="t", spec_type="hold"),
            owner="bob",
            blocking=False,
        )

        queue = await alice_client._open_subscription(
            session_id=sess.session_id, since=0
        )
        try:
            await hub.cancel_task(
                task.task_id, requested_by=alice_client.actor_id, reason="stop"
            )

            from autogen.beta.network.envelope import EV_TASK_CANCELLED

            saw = False
            for _ in range(20):
                try:
                    envelope = await asyncio.wait_for(queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    break
                if (
                    envelope.task_id == task.task_id
                    and envelope.event_type == EV_TASK_CANCELLED
                ):
                    saw = True
                    break
            assert saw
        finally:
            await alice_client._close_subscription(queue)
            release.set()
