# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Blocking vs non-blocking ``Session.create_task`` — Phase 4.

The blocking path subscribes to the session's WAL after the task's create
point, filters by ``task_id``, and resolves on the first terminal task
envelope. These tests exercise every terminal outcome plus the timeout /
already-terminal corner cases.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest
import pytest_asyncio

from autogen.beta.knowledge import MemoryKnowledgeStore
from autogen.beta.network import (
    ActorIdentity,
    Hub,
    HubClient,
    LocalLink,
    Session,
    SessionType,
    Task,
    TaskCancelledError,
    TaskExpiredError,
    TaskFailedError,
    TaskSpec,
    TaskState,
)
from autogen.beta.network.errors import TimeoutError as NetTimeoutError


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
    alice_client = await hub_client.register(
        alice, identity=ActorIdentity(name="alice", capabilities=["ask"])
    )
    bob_client = await hub_client.register(
        bob, identity=ActorIdentity(name="bob", capabilities=["answer"])
    )
    try:
        yield hub, hub_client, alice_client, bob_client
    finally:
        await hub_client.close()
        await link.close()


@pytest_asyncio.fixture
async def session(wired) -> Session:
    _hub, _hc, alice_client, _bob_client = wired
    sess = await alice_client.open(SessionType.CONSULTING, target="bob")
    yield sess


class TestBlockingResolution:
    @pytest.mark.asyncio
    async def test_blocking_returns_terminal_metadata(self, wired, session) -> None:
        _hub, _hc, _alice_client, bob_client = wired

        @bob_client.on_task("one")
        async def _run(envelope, task: Task, client) -> None:  # noqa: ANN001
            await task.result({"answer": 42})

        terminal = await session.create_task(
            TaskSpec(title="compute", spec_type="one"),
            owner="bob",
            blocking=True,
            timeout=2.0,
        )
        assert terminal.state is TaskState.COMPLETED
        assert terminal.result == {"answer": 42}

    @pytest.mark.asyncio
    async def test_blocking_raises_on_failure(self, wired, session) -> None:
        _hub, _hc, _alice_client, bob_client = wired

        @bob_client.on_task("boom")
        async def _run(envelope, task: Task, client) -> None:  # noqa: ANN001
            await task.fail("RuntimeError: timeout after 30s")

        with pytest.raises(TaskFailedError) as excinfo:
            await session.create_task(
                TaskSpec(title="doomed", spec_type="boom"),
                owner="bob",
                blocking=True,
                timeout=2.0,
            )
        err = excinfo.value
        assert "timeout after 30s" in err.reason
        assert err.metadata is not None
        assert err.metadata.state is TaskState.FAILED

    @pytest.mark.asyncio
    async def test_blocking_raises_on_cancellation(self, wired, session) -> None:
        hub, _hc, alice_client, bob_client = wired
        started = asyncio.Event()

        @bob_client.on_task("slow")
        async def _run(envelope, task: Task, client) -> None:  # noqa: ANN001
            await task.phase_entered("work")
            started.set()
            # Sleep long enough that the cancel lands first.
            await asyncio.sleep(5.0)

        async def drive() -> None:
            return await session.create_task(
                TaskSpec(title="held", spec_type="slow"),
                owner="bob",
                blocking=True,
                timeout=3.0,
            )

        blocking_future = asyncio.create_task(drive())
        await asyncio.wait_for(started.wait(), timeout=1.0)
        # Cancel via the hub — simulating a second Alice calling
        # ``task.cancel()`` or ``hub.cancel_task`` directly. We walk
        # the hub's task cache to find the one just created.
        pending = [
            t for t in hub._tasks.values()
            if t.state is TaskState.RUNNING and t.session_id == session.session_id
        ]
        assert len(pending) == 1
        await hub.cancel_task(
            pending[0].task_id, requested_by=alice_client.actor_id, reason="bored"
        )
        with pytest.raises(TaskCancelledError):
            await blocking_future

    @pytest.mark.asyncio
    async def test_blocking_raises_on_timeout(self, wired, session) -> None:
        _hub, _hc, _alice_client, bob_client = wired
        release = asyncio.Event()

        @bob_client.on_task("stall")
        async def _run(envelope, task: Task, client) -> None:  # noqa: ANN001
            await release.wait()

        with pytest.raises(NetTimeoutError):
            await session.create_task(
                TaskSpec(title="stalled", spec_type="stall"),
                owner="bob",
                blocking=True,
                timeout=0.2,
            )
        release.set()

    @pytest.mark.asyncio
    async def test_blocking_raises_on_expiry(self, wired) -> None:
        hub, _hc, alice_client, bob_client = wired
        release = asyncio.Event()

        @bob_client.on_task("slow-ttl")
        async def _run(envelope, task: Task, client) -> None:  # noqa: ANN001
            await task.phase_entered("hold")
            await release.wait()

        sess = await alice_client.open(SessionType.CONSULTING, target="bob")

        async def drive() -> None:
            return await sess.create_task(
                TaskSpec(title="expiring", spec_type="slow-ttl"),
                owner="bob",
                blocking=True,
                timeout=3.0,
                ttl_seconds=1,  # one second TTL
            )

        blocking_future = asyncio.create_task(drive())
        # Give the task time to spin up.
        for _ in range(50):
            tasks = [t for t in hub._tasks.values() if t.state is TaskState.RUNNING]
            if tasks:
                break
            await asyncio.sleep(0.01)
        assert tasks, "task never reached RUNNING"

        # Drive the TTL sweeper forward by passing a clock well past the
        # task's expires_at. This is the deterministic path the design
        # guarantees so tests don't depend on wall-clock time.
        from datetime import datetime, timedelta, timezone

        future = (
            datetime.now(timezone.utc) + timedelta(hours=1)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        await hub.expire_due_tasks(now=future)

        with pytest.raises(TaskExpiredError):
            await blocking_future
        release.set()


class TestNonBlockingHandle:
    @pytest.mark.asyncio
    async def test_returns_handle_immediately(self, wired, session) -> None:
        _hub, _hc, _alice_client, bob_client = wired
        held = asyncio.Event()
        release = asyncio.Event()

        @bob_client.on_task("hold")
        async def _run(envelope, task: Task, client) -> None:  # noqa: ANN001
            held.set()
            await release.wait()
            await task.result("done")

        handle = await session.create_task(
            TaskSpec(title="t", spec_type="hold"),
            owner="bob",
            blocking=False,
        )
        assert isinstance(handle, Task)
        assert not handle.is_terminal()
        assert handle.state is TaskState.CREATED

        await asyncio.wait_for(held.wait(), timeout=1.0)
        # Handle is still non-terminal — handler hasn't resolved yet.
        refreshed = handle.refresh()
        assert refreshed.state in (TaskState.RUNNING, TaskState.CREATED)

        release.set()
        terminal = await handle.wait(timeout=1.0)
        assert terminal.state is TaskState.COMPLETED
        assert terminal.result == "done"

    @pytest.mark.asyncio
    async def test_handle_wait_on_already_terminal(self, wired, session) -> None:
        _hub, _hc, _alice_client, bob_client = wired

        @bob_client.on_task("fast")
        async def _run(envelope, task: Task, client) -> None:  # noqa: ANN001
            await task.result("quick")

        handle = await session.create_task(
            TaskSpec(title="q", spec_type="fast"),
            owner="bob",
            blocking=False,
        )
        # Poll until the handler's emission lands.
        for _ in range(50):
            handle.refresh()
            if handle.is_terminal():
                break
            await asyncio.sleep(0.01)

        terminal = await handle.wait(timeout=0.5)
        assert terminal.state is TaskState.COMPLETED
        assert terminal.result == "quick"
