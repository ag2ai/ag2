# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Phase 4 end-to-end — real framework-core Actor driving network tasks.

These tests wire the full stack: ``autogen.beta.Actor`` with a
``TestConfig`` canned LLM, registered through a real ``HubClient``, running
the default task handler from ``handlers.py``. They prove the "attach an
existing Actor to a hub via the two-client surface" story from the design
doc — the framework-core Actor is not modified, and tasks layer cleanly on
top.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import pytest_asyncio

from autogen.beta import Actor
from autogen.beta.events import ModelMessage, ModelResponse
from autogen.beta.knowledge import DiskKnowledgeStore, MemoryKnowledgeStore
from autogen.beta.network import (
    ActorIdentity,
    Hub,
    HubClient,
    LocalLink,
    SessionType,
    Task,
    TaskPhase,
    TaskSpec,
    TaskState,
)
from autogen.beta.network.envelope import EV_TASK_RESULT, TASK_TERMINAL_EVENT_TYPES
from autogen.beta.testing import TestConfig


class TestDefaultTaskHandler:
    @pytest.mark.asyncio
    async def test_default_handler_bridges_actor_ask_to_result(self) -> None:
        store = MemoryKnowledgeStore()
        hub = Hub(store)
        link = LocalLink()
        link.on_connection(hub.connection_handler)
        hub_client = HubClient(hub, link)
        try:
            alice_cfg = TestConfig(
                ModelResponse(message=ModelMessage(content="[dummy]"))
            )
            # Bob's canned reply is what the default task handler will
            # thread into the TaskResult payload.
            bob_cfg = TestConfig(
                ModelResponse(
                    message=ModelMessage(content="here is your summary: 42")
                )
            )
            alice = Actor("alice", config=alice_cfg)
            bob = Actor("bob", config=bob_cfg)

            alice_client = await hub_client.register(
                alice, identity=ActorIdentity(name="alice")
            )
            await hub_client.register(bob, identity=ActorIdentity(name="bob"))

            sess = await alice_client.open(SessionType.CONSULTING, target="bob")
            terminal = await sess.create_task(
                TaskSpec(
                    title="summarize",
                    description="please summarize the CRISPR 2025 paper",
                ),
                owner="bob",
                blocking=True,
                timeout=3.0,
            )

            assert terminal.state is TaskState.COMPLETED
            assert terminal.result == "here is your summary: 42"
        finally:
            await hub_client.close()
            await link.close()

    @pytest.mark.asyncio
    async def test_default_handler_fails_task_on_exception(self) -> None:
        """Default handler turns a handler exception into ``task.fail``.

        A raised exception inside the handler surfaces as a ``TaskFailedError``
        on ``task.wait()`` — this is the safety net that keeps the blocking
        caller from hanging when the actor's model call blows up.
        """

        class AngryActor:
            name = "angry"

            async def ask(self, content, **kwargs):
                raise RuntimeError("simulated model outage")

        store = MemoryKnowledgeStore()
        hub = Hub(store)
        link = LocalLink()
        link.on_connection(hub.connection_handler)
        hub_client = HubClient(hub, link)
        try:
            caller = Actor(
                "caller",
                config=TestConfig(
                    ModelResponse(message=ModelMessage(content="[dummy]"))
                ),
            )
            caller_client = await hub_client.register(
                caller, identity=ActorIdentity(name="caller")
            )
            await hub_client.register(
                AngryActor(), identity=ActorIdentity(name="angry")
            )
            sess = await caller_client.open(
                SessionType.CONSULTING, target="angry"
            )

            from autogen.beta.network.errors import TaskFailedError

            with pytest.raises(TaskFailedError) as excinfo:
                await sess.create_task(
                    TaskSpec(title="t", description="doit"),
                    owner="angry",
                    blocking=True,
                    timeout=3.0,
                )
            assert "simulated model outage" in excinfo.value.reason
        finally:
            await hub_client.close()
            await link.close()


class _ScriptedActor:
    """Returns responses in order across successive ``ask`` calls.

    Used instead of ``TestConfig`` for multi-call integration tests —
    ``TestConfig.create()`` returns a fresh iterator per call, so a
    framework-core ``Actor`` instantiating the client lazily gets the
    first event every time. A scripted actor tracks the script itself.
    """

    def __init__(self, name: str, script: list[str]) -> None:
        self.name = name
        self._script = list(script)

    async def ask(self, content, **kwargs):
        class _Reply:
            def __init__(self, body: str) -> None:
                self.body = body

        if not self._script:
            return _Reply(f"{self.name}: echo {content}")
        return _Reply(self._script.pop(0))


class TestCustomTaskHandler:
    @pytest.mark.asyncio
    async def test_custom_handler_runs_phased_workflow(self) -> None:
        store = MemoryKnowledgeStore()
        hub = Hub(store)
        link = LocalLink()
        link.on_connection(hub.connection_handler)
        hub_client = HubClient(hub, link)
        try:
            alice = Actor(
                "alice",
                config=TestConfig(
                    ModelResponse(message=ModelMessage(content="[dummy]"))
                ),
            )
            bob = _ScriptedActor(
                "bob", script=["raw search results", "synthesized summary"]
            )

            alice_client = await hub_client.register(
                alice, identity=ActorIdentity(name="alice")
            )
            bob_client = await hub_client.register(
                bob, identity=ActorIdentity(name="bob")
            )

            # Custom handler drives two phases, pulling one model call per
            # phase out of Bob's scripted queue.
            @bob_client.on_task("research")
            async def research(envelope, task: Task, client) -> None:  # noqa: ANN001
                await task.phase_entered("gather", description="search")
                gathered = await client.actor.ask("search for papers")
                await task.phase_completed("gather")
                await task.phase_entered("synthesize", description="summarize")
                summary = await client.actor.ask("synthesize")
                await task.phase_completed("synthesize")
                await task.result(
                    {
                        "raw": gathered.body,
                        "summary": summary.body,
                    }
                )

            sess = await alice_client.open(SessionType.CONSULTING, target="bob")
            terminal = await sess.create_task(
                TaskSpec(
                    title="Literature review",
                    spec_type="research",
                    phases=[TaskPhase(id="gather"), TaskPhase(id="synthesize")],
                ),
                owner="bob",
                blocking=True,
                timeout=3.0,
            )
            assert terminal.state is TaskState.COMPLETED
            assert terminal.result == {
                "raw": "raw search results",
                "summary": "synthesized summary",
            }

            # Both declared phases have both timestamps stamped.
            phases = {p.id: p for p in terminal.spec.phases}
            assert phases["gather"].started_at is not None
            assert phases["gather"].completed_at is not None
            assert phases["synthesize"].started_at is not None
            assert phases["synthesize"].completed_at is not None
        finally:
            await hub_client.close()
            await link.close()

    @pytest.mark.asyncio
    async def test_progress_updates_visible_via_refresh(self) -> None:
        store = MemoryKnowledgeStore()
        hub = Hub(store)
        link = LocalLink()
        link.on_connection(hub.connection_handler)
        hub_client = HubClient(hub, link)
        try:
            alice = Actor(
                "alice",
                config=TestConfig(
                    ModelResponse(message=ModelMessage(content="[dummy]"))
                ),
            )
            bob = Actor(
                "bob",
                config=TestConfig(
                    ModelResponse(message=ModelMessage(content="done")),
                ),
            )
            alice_client = await hub_client.register(
                alice, identity=ActorIdentity(name="alice")
            )
            bob_client = await hub_client.register(
                bob, identity=ActorIdentity(name="bob")
            )

            progress_count = 0

            @bob_client.on_task("crunch")
            async def crunch(envelope, task: Task, client) -> None:  # noqa: ANN001
                nonlocal progress_count
                for i in range(3):
                    await task.progress(step=i + 1, pct=(i + 1) / 3.0)
                    progress_count += 1
                answer = await client.actor.ask("go")
                await task.result(answer.body if hasattr(answer, "body") else str(answer))

            sess = await alice_client.open(SessionType.CONSULTING, target="bob")
            handle = await sess.create_task(
                TaskSpec(title="crunch", spec_type="crunch"),
                owner="bob",
                blocking=False,
            )
            terminal = await handle.wait(timeout=3.0)
            assert progress_count == 3
            assert terminal.state is TaskState.COMPLETED
            # The final metadata captures the most recent progress merge.
            assert terminal.progress["step"] == 3
            assert terminal.progress["pct"] == pytest.approx(1.0)
        finally:
            await hub_client.close()
            await link.close()


class TestTaskDurability:
    @pytest.mark.asyncio
    async def test_task_survives_hub_restart_on_disk_store(
        self, tmp_path: Path
    ) -> None:
        """Disk-backed durability — the full task WAL and metadata round-trip.

        Drives a completed task through the real Actor + handler path on
        a DiskKnowledgeStore, tears the hub and link down, opens a fresh
        :class:`Hub` on the same disk root, and confirms the task is
        still readable with the same terminal state.
        """

        root = tmp_path / "hub"
        store1 = DiskKnowledgeStore(str(root))
        hub1 = Hub(store1)
        link1 = LocalLink()
        link1.on_connection(hub1.connection_handler)
        hc1 = HubClient(hub1, link1)

        alice = Actor(
            "alice",
            config=TestConfig(
                ModelResponse(message=ModelMessage(content="[dummy]"))
            ),
        )
        bob = Actor(
            "bob",
            config=TestConfig(
                ModelResponse(message=ModelMessage(content="persistent-answer"))
            ),
        )
        alice_client = await hc1.register(
            alice, identity=ActorIdentity(name="alice")
        )
        await hc1.register(bob, identity=ActorIdentity(name="bob"))

        sess = await alice_client.open(SessionType.CONSULTING, target="bob")
        terminal = await sess.create_task(
            TaskSpec(title="dur", description="go"),
            owner="bob",
            blocking=True,
            timeout=3.0,
        )
        original_task_id = terminal.task_id
        assert terminal.state is TaskState.COMPLETED

        await hc1.close()
        await link1.close()

        # Fresh hub on the same disk — hydrate reads every task.
        store2 = DiskKnowledgeStore(str(root))
        hub2 = await Hub.open(store2)
        rebuilt = hub2.peek_task(original_task_id)
        assert rebuilt is not None
        assert rebuilt.state is TaskState.COMPLETED
        assert rebuilt.result == "persistent-answer"
