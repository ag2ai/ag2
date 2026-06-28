# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Task durability primitives — cancellation + checkpoint/resume.

The framework-core ``Task`` now supports owner-driven cancellation
and opt-in restart recovery through the :class:`CheckpointStore`
Protocol. The hub ships :class:`HubBackedCheckpointStore` as the
canonical default; tenants may plug in any compatible store.
"""

import asyncio

import pytest

from ag2 import Agent, Context
from ag2.events import TaskCancelled
from ag2.knowledge import MemoryKnowledgeStore
from ag2.network import (
    Hub,
    HubBackedCheckpointStore,
    Passport,
    Resume,
)
from ag2.network.task_mirror import TaskMirror
from ag2.stream import MemoryStream
from ag2.task import (
    TERMINAL_TASK_STATES,
    CheckpointStore,
    Task,
    TaskSpec,
    TaskState,
)
from ag2.testing import TestConfig


class _InMemoryCheckpointStore:
    """Test double satisfying :class:`CheckpointStore` without a hub."""

    def __init__(self) -> None:
        self.data: dict[str, dict] = {}

    async def write(self, task_id: str, state: dict) -> None:
        self.data[task_id] = dict(state)

    async def read(self, task_id: str) -> dict | None:
        snap = self.data.get(task_id)
        return dict(snap) if snap is not None else None


class TestCancellationState:
    def test_cancelled_is_a_terminal_state(self) -> None:
        assert TaskState.CANCELLED in TERMINAL_TASK_STATES

    def test_task_state_values_include_cancelled(self) -> None:
        assert TaskState.CANCELLED.value == "cancelled"


class TestTaskCancel:
    @pytest.mark.asyncio
    async def test_cancel_transitions_to_cancelled_and_emits_event(self) -> None:
        stream = MemoryStream()
        events: list = []
        stream.subscribe(lambda ev: events.append(ev))

        from ag2.context import ConversationContext

        task = Task(
            owner_id="alice",
            spec=TaskSpec(title="thing"),
            context=ConversationContext(stream=stream),
        )
        async with task:
            await task.cancel("ran too long")
            assert task.state == TaskState.CANCELLED
            assert task.metadata.error == "ran too long"

        # Captured TaskCancelled with the reason.
        cancels = [e for e in events if isinstance(e, TaskCancelled)]
        assert len(cancels) == 1
        assert cancels[0].reason == "ran too long"

    @pytest.mark.asyncio
    async def test_cancel_is_idempotent_on_terminal_task(self) -> None:
        from ag2.context import ConversationContext

        stream = MemoryStream()
        events: list = []
        stream.subscribe(lambda ev: events.append(ev))

        task = Task(
            owner_id="alice",
            spec=TaskSpec(title="thing"),
            context=ConversationContext(stream=stream),
        )
        async with task:
            await task.complete("done")
            assert task.state == TaskState.COMPLETED
            # Second terminal call must not flip the state or emit another event.
            await task.cancel("late")
            assert task.state == TaskState.COMPLETED

        assert not any(isinstance(e, TaskCancelled) for e in events)

    @pytest.mark.asyncio
    async def test_cancel_via_agent_task_helper(self) -> None:
        agent = Agent(name="alice", config=TestConfig())
        async with agent.task("work") as task:
            await task.cancel()
            assert task.state == TaskState.CANCELLED


class TestCheckpointStandalone:
    @pytest.mark.asyncio
    async def test_checkpoint_writes_via_store_and_resume_reads_it_back(self) -> None:
        store = _InMemoryCheckpointStore()

        from ag2.context import ConversationContext

        first = Task(
            owner_id="alice",
            spec=TaskSpec(title="work"),
            context=ConversationContext(stream=MemoryStream()),
            checkpoint_store=store,
        )
        prior_task_id: str
        async with first:
            await first.checkpoint({"step": 3, "scratch": [1, 2, 3]})
            prior_task_id = first.task_id

        second = Task(
            owner_id="alice",
            spec=TaskSpec(title="work"),
            context=ConversationContext(stream=MemoryStream()),
            checkpoint_store=store,
            resume_from=prior_task_id,
        )
        async with second:
            assert second.resumed_state == {"step": 3, "scratch": [1, 2, 3]}

    @pytest.mark.asyncio
    async def test_checkpoint_without_store_is_a_silent_noop(self) -> None:
        """Standalone agents that never wire a store can still call
        ``Task.checkpoint`` — the call is just dropped."""
        from ag2.context import ConversationContext

        task = Task(
            owner_id="alice",
            spec=TaskSpec(title="work"),
            context=ConversationContext(stream=MemoryStream()),
        )
        async with task:
            await task.checkpoint({"step": 1})  # must not raise

    @pytest.mark.asyncio
    async def test_resume_from_unknown_task_yields_none(self) -> None:
        store = _InMemoryCheckpointStore()
        from ag2.context import ConversationContext

        task = Task(
            owner_id="alice",
            spec=TaskSpec(title="work"),
            context=ConversationContext(stream=MemoryStream()),
            checkpoint_store=store,
            resume_from="never-checkpointed",
        )
        async with task:
            assert task.resumed_state is None

    @pytest.mark.asyncio
    async def test_checkpoint_after_terminal_is_a_noop(self) -> None:
        store = _InMemoryCheckpointStore()
        from ag2.context import ConversationContext

        task = Task(
            owner_id="alice",
            spec=TaskSpec(title="work"),
            context=ConversationContext(stream=MemoryStream()),
            checkpoint_store=store,
        )
        async with task:
            tid = task.task_id
            await task.complete("done")
            await task.checkpoint({"step": "should-not-persist"})

        assert tid not in store.data


class TestHubBackedCheckpointStore:
    @pytest.mark.asyncio
    async def test_round_trip_through_hub(self) -> None:
        hub = await Hub.open(
            MemoryKnowledgeStore(),
            ttl_sweep_interval=0,
            expectation_sweep_interval=0,
        )
        try:
            store = HubBackedCheckpointStore(hub)
            await store.write("task-42", {"step": 7, "buffer": ["a", "b"]})
            assert await store.read("task-42") == {"step": 7, "buffer": ["a", "b"]}
        finally:
            await hub.close()

    @pytest.mark.asyncio
    async def test_read_unknown_task_returns_none(self) -> None:
        hub = await Hub.open(
            MemoryKnowledgeStore(),
            ttl_sweep_interval=0,
            expectation_sweep_interval=0,
        )
        try:
            store = HubBackedCheckpointStore(hub)
            assert await store.read("never-written") is None
        finally:
            await hub.close()

    @pytest.mark.asyncio
    async def test_last_write_wins(self) -> None:
        hub = await Hub.open(
            MemoryKnowledgeStore(),
            ttl_sweep_interval=0,
            expectation_sweep_interval=0,
        )
        try:
            store = HubBackedCheckpointStore(hub)
            await store.write("task-1", {"v": 1})
            await store.write("task-1", {"v": 2})
            assert await store.read("task-1") == {"v": 2}
        finally:
            await hub.close()

    @pytest.mark.asyncio
    async def test_full_resume_cycle_through_hub(self) -> None:
        """End-to-end: first run checkpoints via hub-backed store; a
        fresh task that resumes from the prior id sees the snapshot."""
        hub = await Hub.open(
            MemoryKnowledgeStore(),
            ttl_sweep_interval=0,
            expectation_sweep_interval=0,
        )
        try:
            store = HubBackedCheckpointStore(hub)
            agent = Agent(name="alice", config=TestConfig())

            async with agent.task("work", checkpoint_store=store) as first:
                await first.checkpoint({"phase": "research", "found": 42})
                prior_id = first.task_id

            async with agent.task("work", checkpoint_store=store, resume_from=prior_id) as second:
                assert second.resumed_state == {"phase": "research", "found": 42}
                await second.complete("resumed and finished")
                assert second.state == TaskState.COMPLETED
        finally:
            await hub.close()

    def test_protocol_conformance_runtime_checkable(self) -> None:
        # CheckpointStore is runtime_checkable; HubBackedCheckpointStore
        # should satisfy isinstance even without an actual hub instance.
        # Use a sentinel object for the hub param since the constructor
        # only stores it without using it during the isinstance check.
        store = HubBackedCheckpointStore.__new__(HubBackedCheckpointStore)
        assert isinstance(store, CheckpointStore)

    def test_in_memory_double_also_satisfies_protocol(self) -> None:
        assert isinstance(_InMemoryCheckpointStore(), CheckpointStore)


@pytest.mark.asyncio
class TestMirrorCancellation:
    """``TaskMirror`` forwards owner-driven cancellation to the hub.

    Without this bridge, an agent calling ``Task.cancel`` would emit a
    ``TaskCancelled`` event on its stream but the hub's task cache
    would stay at ``RUNNING`` and capability-tagged stats would never
    record the cancellation.
    """

    async def test_mirror_forwards_cancellation_to_hub_metadata(self) -> None:
        hub = await Hub.open(
            MemoryKnowledgeStore(),
            ttl_sweep_interval=0,
            expectation_sweep_interval=0,
        )
        try:
            bob_passport = await hub.register_identity(Passport(name="bob"), Resume())
            agent = Agent(name="bob", config=TestConfig())
            stream = MemoryStream()
            mirror = TaskMirror(hub=hub, owner_id=bob_passport.agent_id)
            sub_ids = mirror.attach(stream)
            try:
                async with agent.task("indexing", context=Context(stream=stream)) as task:
                    task_id = task.task_id
                    await task.cancel("wrap-up")
                # Stream callbacks run via subscribe(sync_to_thread=False) — yield
                # the loop so the mirror's _on_cancelled completes before assert.
                await asyncio.sleep(0)
            finally:
                mirror.detach(stream, sub_ids)

            meta = await hub.get_task(task_id)
            assert meta.state == TaskState.CANCELLED
            assert meta.error == "wrap-up"
            assert meta.completed_at  # terminal stamp
        finally:
            await hub.close()

    async def test_mirror_records_observation_on_capability_tagged_cancellation(
        self,
    ) -> None:
        hub = await Hub.open(
            MemoryKnowledgeStore(),
            ttl_sweep_interval=0,
            expectation_sweep_interval=0,
        )
        try:
            bob_passport = await hub.register_identity(
                Passport(name="bob"),
                Resume(claimed_capabilities=["indexing"]),
            )
            agent = Agent(name="bob", config=TestConfig())
            stream = MemoryStream()
            mirror = TaskMirror(hub=hub, owner_id=bob_passport.agent_id)
            sub_ids = mirror.attach(stream)
            try:
                async with agent.task(
                    "indexing",
                    description="search & summarise",
                    capability="indexing",
                    context=Context(stream=stream),
                ) as task:
                    await task.cancel("wrap-up")
                await asyncio.sleep(0)
            finally:
                mirror.detach(stream, sub_ids)

            resume = await hub.get_resume(bob_passport.agent_id)
            stat = resume.observed.get("indexing")
            assert stat is not None
            assert stat.n == 1
        finally:
            await hub.close()


@pytest.mark.asyncio
class TestHubAccessors:
    """``Hub.find_agent_id`` + ``Hub.get_rule`` are the public-surface
    primitives ``HubClient.attach`` uses to look up persisted identity
    without reaching into hub internals.
    """

    async def test_find_agent_id_returns_id_for_registered_name(self) -> None:
        hub = await Hub.open(
            MemoryKnowledgeStore(),
            ttl_sweep_interval=0,
            expectation_sweep_interval=0,
        )
        try:
            passport = await hub.register_identity(Passport(name="alice"), Resume())
            assert hub.find_agent_id("alice") == passport.agent_id
        finally:
            await hub.close()

    async def test_find_agent_id_returns_none_for_unknown_name(self) -> None:
        hub = await Hub.open(
            MemoryKnowledgeStore(),
            ttl_sweep_interval=0,
            expectation_sweep_interval=0,
        )
        try:
            assert hub.find_agent_id("ghost") is None
        finally:
            await hub.close()

    async def test_get_rule_returns_default_rule_for_registered_agent(self) -> None:
        from ag2.network import Rule

        hub = await Hub.open(
            MemoryKnowledgeStore(),
            ttl_sweep_interval=0,
            expectation_sweep_interval=0,
        )
        try:
            passport = await hub.register_identity(Passport(name="alice"), Resume())
            rule = await hub.get_rule(passport.agent_id)
            assert isinstance(rule, Rule)
        finally:
            await hub.close()

    async def test_get_rule_raises_not_found_for_unknown_agent(self) -> None:
        from ag2.network import NotFoundError

        hub = await Hub.open(
            MemoryKnowledgeStore(),
            ttl_sweep_interval=0,
            expectation_sweep_interval=0,
        )
        try:
            with pytest.raises(NotFoundError):
                await hub.get_rule("never-registered")
        finally:
            await hub.close()
