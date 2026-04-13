# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Cold-restart hydrate for tasks — Phase 4.

A hub process that crashes mid-task should be able to rebuild its task
cache from ``hub/tasks/*/metadata.json`` without losing state. Running
tasks stay ``running`` (the TTL sweeper is the safety net); terminal
tasks stay terminal.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import pytest
import pytest_asyncio

from autogen.beta.knowledge import DiskKnowledgeStore, MemoryKnowledgeStore
from autogen.beta.network import (
    ActorIdentity,
    Hub,
    HubClient,
    LocalLink,
    SessionType,
    Task,
    TaskSpec,
    TaskState,
)
from autogen.beta.network.hub import layout
from autogen.beta.network.task import TaskMetadata


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


class TestHydrateTasks:
    @pytest.mark.asyncio
    async def test_hydrate_rebuilds_completed_task(self, wired) -> None:
        _hub, store, _hc, alice_client, bob_client = wired

        @bob_client.on_task("fast")
        async def _run(envelope, task: Task, client) -> None:  # noqa: ANN001
            await task.result({"ok": True})

        sess = await alice_client.open(SessionType.CONSULTING, target="bob")
        original = await sess.create_task(
            TaskSpec(title="t", spec_type="fast"),
            owner="bob",
            blocking=True,
            timeout=2.0,
        )
        assert original.state is TaskState.COMPLETED

        # Fresh hub pointed at the same store — simulate cold restart.
        new_hub = await Hub.open(store)

        # Tasks cache rebuilt
        rebuilt = new_hub.peek_task(original.task_id)
        assert rebuilt is not None
        assert rebuilt.state is TaskState.COMPLETED
        assert rebuilt.result == {"ok": True}
        # And discoverable through the session lookup helper.
        by_session = new_hub.tasks_for_session(original.session_id)
        assert len(by_session) == 1
        assert by_session[0].task_id == original.task_id

    @pytest.mark.asyncio
    async def test_hydrate_rebuilds_running_task_preserving_phase(
        self, wired
    ) -> None:
        _hub, store, _hc, alice_client, bob_client = wired
        release = asyncio.Event()

        @bob_client.on_task("long")
        async def _hold(envelope, task: Task, client) -> None:  # noqa: ANN001
            await task.phase_entered("gather")
            await release.wait()

        sess = await alice_client.open(SessionType.CONSULTING, target="bob")
        live = await sess.create_task(
            TaskSpec(title="t", spec_type="long"),
            owner="bob",
            blocking=False,
        )
        # Wait for the phase-entered event to reach the hub cache.
        for _ in range(50):
            peek = _hub.peek_task(live.task_id)
            if peek is not None and peek.current_phase == "gather":
                break
            await asyncio.sleep(0.01)

        # Cold restart the hub.
        new_hub = await Hub.open(store)
        rebuilt = new_hub.peek_task(live.task_id)
        assert rebuilt is not None
        assert rebuilt.state is TaskState.RUNNING
        assert rebuilt.current_phase == "gather"
        assert rebuilt.started_at is not None

        # Non-terminal tasks get added back to the session→task index
        # so ``Session.track_tasks`` / ``Hub.tasks_for_session`` return
        # them on subsequent requests.
        assert live.task_id in new_hub._session_tasks.get(sess.session_id, set())
        release.set()

    @pytest.mark.asyncio
    async def test_hydrate_partial_metadata_skipped(self, wired) -> None:
        _hub, store, _hc, _alice_client, _bob_client = wired

        # Write a corrupt metadata file under hub/tasks/<id>.
        bad_id = "01932fakebad"
        await store.write(
            layout.task_metadata(bad_id),
            "{not valid json",
        )

        # Hydrate should log + skip without raising.
        new_hub = await Hub.open(store)
        assert new_hub.peek_task(bad_id) is None


class TestHydrateDisk:
    @pytest.mark.asyncio
    async def test_hydrate_round_trip_on_disk_store(self, tmp_path) -> None:
        store = DiskKnowledgeStore(str(tmp_path / "store"))
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

        @bob_client.on_task("fast")
        async def _run(envelope, task: Task, client) -> None:  # noqa: ANN001
            await task.result("persisted")

        sess = await alice_client.open(SessionType.CONSULTING, target="bob")
        original = await sess.create_task(
            TaskSpec(title="t", spec_type="fast"),
            owner="bob",
            blocking=True,
            timeout=2.0,
        )
        await hub_client.close()
        await link.close()

        # Fresh hub on the same disk store.
        store2 = DiskKnowledgeStore(str(tmp_path / "store"))
        new_hub = await Hub.open(store2)
        rebuilt = new_hub.peek_task(original.task_id)
        assert rebuilt is not None
        assert rebuilt.state is TaskState.COMPLETED
        assert rebuilt.result == "persisted"

        # The metadata file was actually written — confirm by reading
        # it through the store directly.
        raw = await store2.read(layout.task_metadata(original.task_id))
        assert raw is not None
        parsed = TaskMetadata.from_json(raw)
        assert parsed.state is TaskState.COMPLETED
