# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Task durability primitives — cancellation + checkpoint/resume.

The framework-core ``Task`` now supports owner-driven cancellation
and opt-in restart recovery through the :class:`CheckpointStore`
Protocol. The hub ships :class:`HubBackedCheckpointStore` as the
canonical default; tenants may plug in any compatible store.
"""

import pytest

from autogen.beta import Agent
from autogen.beta.events import TaskCancelled
from autogen.beta.knowledge import MemoryKnowledgeStore
from autogen.beta.network import (
    Hub,
    HubBackedCheckpointStore,
)
from autogen.beta.stream import MemoryStream
from autogen.beta.task import (
    TERMINAL_TASK_STATES,
    CheckpointStore,
    Task,
    TaskSpec,
    TaskState,
)
from autogen.beta.testing import TestConfig


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

        from autogen.beta.context import ConversationContext

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
        from autogen.beta.context import ConversationContext

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

        from autogen.beta.context import ConversationContext

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
        from autogen.beta.context import ConversationContext

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
        from autogen.beta.context import ConversationContext

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
        from autogen.beta.context import ConversationContext

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

            async with agent.task(
                "work", checkpoint_store=store, resume_from=prior_id
            ) as second:
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
