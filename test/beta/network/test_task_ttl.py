# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Task TTL sweeper — Phase 4.

``Hub.expire_due_tasks`` is the deterministic entry point the sweeper's
``IntervalWatch`` callback invokes. Tests drive it manually with a
future-dated clock so we never depend on real time.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
import pytest_asyncio

from autogen.beta.knowledge import MemoryKnowledgeStore
from autogen.beta.network import (
    AccessBlock,
    ActorIdentity,
    Hub,
    HubClient,
    LimitsBlock,
    LocalLink,
    Rule,
    SessionType,
    Task,
    TaskSpec,
    TaskState,
)
from autogen.beta.network.hub import layout


@dataclass
class FakeReply:
    content: str


class FakeActor:
    def __init__(self, name: str) -> None:
        self.name = name

    async def ask(self, content: str, **kwargs: Any) -> FakeReply:
        return FakeReply(content=f"{self.name}: {content}")


def future_iso(seconds: int = 3600) -> str:
    return (
        datetime.now(timezone.utc) + timedelta(seconds=seconds)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")


@pytest_asyncio.fixture
async def wired():
    store = MemoryKnowledgeStore()
    hub = Hub(store)
    link = LocalLink()
    link.on_connection(hub.connection_handler)
    hub_client = HubClient(hub, link)

    alice = FakeActor(name="alice")
    bob = FakeActor(name="bob")
    alice_client = await hub_client.register(
        alice, identity=ActorIdentity(name="alice")
    )
    bob_client = await hub_client.register(
        bob, identity=ActorIdentity(name="bob")
    )
    try:
        yield hub, store, hub_client, alice_client, bob_client
    finally:
        await hub_client.close()
        await link.close()


class TestExpireDueTasks:
    @pytest.mark.asyncio
    async def test_expires_non_terminal_task_past_deadline(self, wired) -> None:
        hub, store, _hc, alice_client, bob_client = wired
        release = asyncio.Event()

        @bob_client.on_task("hold")
        async def _hold(envelope, task: Task, client) -> None:  # noqa: ANN001
            await task.phase_entered("wait")
            await release.wait()

        sess = await alice_client.open(SessionType.CONSULTING, target="bob")
        task = await sess.create_task(
            TaskSpec(title="t", spec_type="hold"),
            owner="bob",
            blocking=False,
            ttl_seconds=1,
        )

        # Spin until the handler has picked up the task.
        for _ in range(50):
            if hub.peek_task(task.task_id).state is TaskState.RUNNING:
                break
            await asyncio.sleep(0.01)

        expired_ids = await hub.expire_due_tasks(now=future_iso(3600))
        assert task.task_id in expired_ids

        refreshed = hub.peek_task(task.task_id)
        assert refreshed.state is TaskState.EXPIRED
        assert refreshed.completed_at is not None

        # Durable metadata reflects the expiry.
        raw = await store.read(layout.task_metadata(task.task_id))
        assert json.loads(raw)["state"] == "expired"
        release.set()

    @pytest.mark.asyncio
    async def test_skips_task_not_yet_due(self, wired) -> None:
        hub, _store, _hc, alice_client, bob_client = wired
        release = asyncio.Event()

        @bob_client.on_task("hold")
        async def _hold(envelope, task: Task, client) -> None:  # noqa: ANN001
            await release.wait()

        sess = await alice_client.open(SessionType.CONSULTING, target="bob")
        task = await sess.create_task(
            TaskSpec(title="t", spec_type="hold"),
            owner="bob",
            blocking=False,
            ttl_seconds=3600,  # one-hour TTL
        )

        # Sweep with the current clock — nothing should expire.
        expired_ids = await hub.expire_due_tasks()
        assert expired_ids == []
        assert hub.peek_task(task.task_id).state in (
            TaskState.CREATED,
            TaskState.RUNNING,
        )
        release.set()

    @pytest.mark.asyncio
    async def test_skips_terminal_tasks(self, wired) -> None:
        hub, _store, _hc, alice_client, bob_client = wired

        @bob_client.on_task("fast")
        async def _fast(envelope, task: Task, client) -> None:  # noqa: ANN001
            await task.result("done")

        sess = await alice_client.open(SessionType.CONSULTING, target="bob")
        task = await sess.create_task(
            TaskSpec(title="t", spec_type="fast"),
            owner="bob",
            blocking=True,
            timeout=2.0,
            ttl_seconds=1,
        )
        assert task.state is TaskState.COMPLETED

        # Expiry sweeper must not re-expire a completed task even
        # though its ``expires_at`` is in the past.
        expired_ids = await hub.expire_due_tasks(now=future_iso(3600))
        assert expired_ids == []
        assert hub.peek_task(task.task_id).state is TaskState.COMPLETED

    @pytest.mark.asyncio
    async def test_expiry_fans_out_to_subscribers(self, wired) -> None:
        hub, _store, _hc, alice_client, bob_client = wired
        release = asyncio.Event()

        @bob_client.on_task("hold")
        async def _hold(envelope, task: Task, client) -> None:  # noqa: ANN001
            await release.wait()

        sess = await alice_client.open(SessionType.CONSULTING, target="bob")
        task = await sess.create_task(
            TaskSpec(title="t", spec_type="hold"),
            owner="bob",
            blocking=False,
            ttl_seconds=1,
        )

        # Open a subscription from Alice's side on the whole session.
        queue = await alice_client._open_subscription(
            session_id=sess.session_id, since=0
        )
        try:
            await hub.expire_due_tasks(now=future_iso(3600))

            from autogen.beta.network.envelope import EV_TASK_EXPIRED

            saw_expired = False
            for _ in range(20):
                try:
                    envelope = await asyncio.wait_for(queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    break
                if (
                    envelope.task_id == task.task_id
                    and envelope.event_type == EV_TASK_EXPIRED
                ):
                    saw_expired = True
                    break
            assert saw_expired
        finally:
            await alice_client._close_subscription(queue)
            release.set()

    @pytest.mark.asyncio
    async def test_multi_session_isolation(self, wired) -> None:
        hub, _store, _hc, alice_client, bob_client = wired
        release = asyncio.Event()

        @bob_client.on_task("hold")
        async def _hold(envelope, task: Task, client) -> None:  # noqa: ANN001
            await release.wait()

        sess1 = await alice_client.open(SessionType.CONSULTING, target="bob")
        short = await sess1.create_task(
            TaskSpec(title="t1", spec_type="hold"),
            owner="bob",
            blocking=False,
            ttl_seconds=1,
        )

        sess2 = await alice_client.open(SessionType.CONSULTING, target="bob")
        long = await sess2.create_task(
            TaskSpec(title="t2", spec_type="hold"),
            owner="bob",
            blocking=False,
            ttl_seconds=3600,
        )

        expired = await hub.expire_due_tasks(now=future_iso(60))
        assert short.task_id in expired
        assert long.task_id not in expired

        assert hub.peek_task(short.task_id).state is TaskState.EXPIRED
        assert hub.peek_task(long.task_id).state in (
            TaskState.CREATED,
            TaskState.RUNNING,
        )
        release.set()
