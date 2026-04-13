# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Task lifecycle end-to-end — Phase 4.

Wires a real Hub + LocalLink + two :class:`FakeActor` identities through
:class:`HubClient` / :class:`ActorClient` / :class:`Session`, and drives:

* ``Session.create_task`` → ``ag2.task.assigned`` envelope delivered to the
  owner via the task-handler registry,
* owner emits ``phase_entered`` / ``progress`` / ``phase_completed`` / ``result``
  through the :class:`Task` handle,
* hub's ``_tasks`` cache and on-disk ``hub/tasks/{id}/metadata.json`` stay
  consistent with the WAL,
* consulting 1Q1R does **not** trip — the session stays ACTIVE while tasks
  emit freely.
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
    AccessBlock,
    ActorClient,
    ActorIdentity,
    Hub,
    HubClient,
    LimitsBlock,
    LocalLink,
    Rule,
    Session,
    SessionState,
    SessionType,
    Task,
    TaskPhase,
    TaskSpec,
    TaskState,
)
from autogen.beta.network.hub import layout


# ---------------------------------------------------------------------------
# Test actor
# ---------------------------------------------------------------------------


@dataclass
class FakeReply:
    content: str


class FakeActor:
    def __init__(self, name: str, reply: str = "") -> None:
        self.name = name
        self.reply = reply
        self.questions: list[str] = []

    async def ask(self, content: str, **kwargs: Any) -> FakeReply:
        self.questions.append(content)
        return FakeReply(content=self.reply or f"{self.name}: {content}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def wired():
    """A hub + two registered clients ready to create sessions."""

    store = MemoryKnowledgeStore()
    hub = Hub(store)
    link = LocalLink()
    link.on_connection(hub.connection_handler)
    hub_client = HubClient(hub, link)

    alice = FakeActor(name="alice", reply="alice-reply")
    bob = FakeActor(name="bob", reply="bob-reply")
    alice_client = await hub_client.register(
        alice, identity=ActorIdentity(name="alice", capabilities=["ask"])
    )
    bob_client = await hub_client.register(
        bob, identity=ActorIdentity(name="bob", capabilities=["answer"])
    )

    try:
        yield hub, store, hub_client, alice_client, bob_client, alice, bob
    finally:
        await hub_client.close()
        await link.close()


@pytest_asyncio.fixture
async def session(wired) -> Session:
    _hub, _store, _hc, alice_client, _bob_client, _alice, _bob = wired
    sess = await alice_client.open(SessionType.CONSULTING, target="bob")
    yield sess


# ---------------------------------------------------------------------------
# Hub.create_task — happy path
# ---------------------------------------------------------------------------


class TestCreateTask:
    @pytest.mark.asyncio
    async def test_allocates_task_and_metadata_on_disk(self, wired, session) -> None:
        hub, store, _hc, _alice_client, bob_client, _alice, _bob = wired

        # Register an owner-side handler on Bob's client that captures
        # the assignment but doesn't immediately finish the task — we
        # want to inspect the freshly-created task in a deterministic
        # state rather than race the default handler.
        assignment_seen = asyncio.Event()
        captured: dict[str, Task] = {}

        @bob_client.on_task("*")
        async def _hold_task(envelope, task: Task, client) -> None:  # noqa: ANN001
            captured["task"] = task
            assignment_seen.set()

        spec = TaskSpec(
            title="Summarize the papers",
            description="produce a 500-word summary",
            phases=[TaskPhase(id="fetch"), TaskPhase(id="write")],
            spec_type="research",
        )

        task = await session.create_task(spec, owner="bob", blocking=False)
        await asyncio.wait_for(assignment_seen.wait(), timeout=1.0)

        assert isinstance(task, Task)
        assert task.metadata.state is TaskState.CREATED
        assert task.metadata.owner_id == bob_client.actor_id
        assert task.metadata.spec.title == "Summarize the papers"
        assert [p.id for p in task.metadata.spec.phases] == ["fetch", "write"]

        # Hub-side cache
        cached = hub.peek_task(task.task_id)
        assert cached is not None
        assert cached.state is TaskState.CREATED
        assert cached.session_id == session.session_id

        # Durable metadata on disk
        raw = await store.read(layout.task_metadata(task.task_id))
        assert raw is not None
        on_disk = json.loads(raw)
        assert on_disk["task_id"] == task.task_id
        assert on_disk["state"] == "created"
        assert on_disk["spec"]["title"] == "Summarize the papers"

        # Session-side back-reference
        ref_raw = await store.read(
            layout.session_task_ref(session.session_id, task.task_id)
        )
        assert ref_raw == task.task_id

        # The owner's handler saw the same handle as what create_task
        # returned (different instances but same underlying task).
        assert captured["task"].task_id == task.task_id

    @pytest.mark.asyncio
    async def test_rejects_unknown_owner(self, wired, session) -> None:
        _hub, _store, _hc, _alice_client, _bob_client, _alice, _bob = wired
        with pytest.raises(ValueError):
            await session.create_task(
                TaskSpec(title="t"), owner="ghost", blocking=False
            )

    @pytest.mark.asyncio
    async def test_rejects_non_participant_owner(self, wired) -> None:
        hub, _store, hub_client, alice_client, bob_client, _alice, _bob = wired
        carol = FakeActor(name="carol")
        await hub_client.register(
            carol, identity=ActorIdentity(name="carol", capabilities=["ask"])
        )
        sess = await alice_client.open(SessionType.CONSULTING, target="bob")
        with pytest.raises(ValueError):
            await sess.create_task(
                TaskSpec(title="t"), owner="carol", blocking=False
            )

    @pytest.mark.asyncio
    async def test_enforces_max_concurrent_tasks(self, wired) -> None:
        hub, _store, hub_client, alice_client, _bob_client, _alice, _bob = wired
        carol = FakeActor(name="carol")
        # Carol has a hard 1-concurrent-task ceiling.
        carol_rule = Rule(
            access=AccessBlock(),
            limits=LimitsBlock(max_concurrent_tasks=1, task_ttl_default="1h"),
        )
        carol_client = await hub_client.register(
            carol,
            identity=ActorIdentity(name="carol", capabilities=["work"]),
            rule=carol_rule,
        )

        # A slow handler pins Carol's one task slot without finishing.
        release = asyncio.Event()

        @carol_client.on_task("slow")
        async def _hold(envelope, task: Task, client) -> None:  # noqa: ANN001
            await task.phase_entered("work")
            await release.wait()

        sess = await alice_client.open(SessionType.CONSULTING, target="carol")
        task1 = await sess.create_task(
            TaskSpec(
                title="t1",
                description="hold",
                spec_type="slow",
                phases=[TaskPhase(id="work")],
            ),
            owner="carol",
            blocking=False,
        )

        # Wait until Carol's handler has actually picked the task up,
        # otherwise ``max_concurrent_tasks`` still has a free slot.
        for _ in range(50):
            t1 = hub.peek_task(task1.task_id)
            if t1 is not None and t1.state is TaskState.RUNNING:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("task1 never reached RUNNING")

        from autogen.beta.network.errors import LimitExceededError

        with pytest.raises(LimitExceededError):
            await sess.create_task(
                TaskSpec(title="t2", description="hold2", spec_type="slow"),
                owner="carol",
                blocking=False,
            )
        release.set()


# ---------------------------------------------------------------------------
# Task event lifecycle — owner-side emissions
# ---------------------------------------------------------------------------


class TestTaskEmissions:
    @pytest.mark.asyncio
    async def test_phase_entered_transitions_to_running(self, wired, session) -> None:
        hub, _store, _hc, _alice_client, bob_client, _alice, _bob = wired

        started = asyncio.Event()
        owner_task_ref: dict[str, Task] = {}

        @bob_client.on_task("phases")
        async def _run(envelope, task: Task, client) -> None:  # noqa: ANN001
            owner_task_ref["task"] = task
            await task.phase_entered("outline", description="rough plan")
            started.set()

        task = await session.create_task(
            TaskSpec(
                title="write",
                spec_type="phases",
                phases=[TaskPhase(id="outline"), TaskPhase(id="draft")],
            ),
            owner="bob",
            blocking=False,
        )
        await asyncio.wait_for(started.wait(), timeout=1.0)

        updated = hub.peek_task(task.task_id)
        assert updated is not None
        assert updated.state is TaskState.RUNNING
        assert updated.current_phase == "outline"
        assert updated.started_at is not None
        # The declared phase has its ``started_at`` stamped.
        outline = next(p for p in updated.spec.phases if p.id == "outline")
        assert outline.started_at is not None

    @pytest.mark.asyncio
    async def test_progress_merges_without_terminal(self, wired, session) -> None:
        hub, _store, _hc, _alice_client, bob_client, _alice, _bob = wired

        progressed = asyncio.Event()

        @bob_client.on_task("crunch")
        async def _run(envelope, task: Task, client) -> None:  # noqa: ANN001
            await task.progress(docs=2, pct=0.25)
            await task.progress(docs=4, pct=0.50)
            progressed.set()

        task = await session.create_task(
            TaskSpec(title="crunch", spec_type="crunch"),
            owner="bob",
            blocking=False,
        )
        await asyncio.wait_for(progressed.wait(), timeout=1.0)

        updated = hub.peek_task(task.task_id)
        assert updated is not None
        assert updated.state is TaskState.RUNNING
        assert updated.progress == {"docs": 4, "pct": 0.50}
        assert updated.last_progress_at is not None

    @pytest.mark.asyncio
    async def test_result_transitions_to_completed(self, wired, session) -> None:
        hub, _store, _hc, _alice_client, bob_client, _alice, _bob = wired

        done = asyncio.Event()

        @bob_client.on_task("oneshot")
        async def _run(envelope, task: Task, client) -> None:  # noqa: ANN001
            await task.result({"answer": 42})
            done.set()

        task = await session.create_task(
            TaskSpec(title="compute", spec_type="oneshot"),
            owner="bob",
            blocking=False,
        )
        await asyncio.wait_for(done.wait(), timeout=1.0)
        # Give the hub one tick to process the final ack.
        await asyncio.sleep(0)

        updated = hub.peek_task(task.task_id)
        assert updated is not None
        assert updated.state is TaskState.COMPLETED
        assert updated.result == {"answer": 42}
        assert updated.completed_at is not None

    @pytest.mark.asyncio
    async def test_error_transitions_to_failed(self, wired, session) -> None:
        hub, _store, _hc, _alice_client, bob_client, _alice, _bob = wired

        errored = asyncio.Event()

        @bob_client.on_task("boom")
        async def _run(envelope, task: Task, client) -> None:  # noqa: ANN001
            await task.fail("RuntimeError: model timed out")
            errored.set()

        task = await session.create_task(
            TaskSpec(title="crash", spec_type="boom"),
            owner="bob",
            blocking=False,
        )
        await asyncio.wait_for(errored.wait(), timeout=1.0)
        await asyncio.sleep(0)

        updated = hub.peek_task(task.task_id)
        assert updated is not None
        assert updated.state is TaskState.FAILED
        assert updated.error == "RuntimeError: model timed out"

    @pytest.mark.asyncio
    async def test_phase_completed_stamps_phase_timestamp(
        self, wired, session
    ) -> None:
        hub, _store, _hc, _alice_client, bob_client, _alice, _bob = wired
        done = asyncio.Event()

        @bob_client.on_task("plan")
        async def _run(envelope, task: Task, client) -> None:  # noqa: ANN001
            await task.phase_entered("plan-it")
            await task.phase_completed("plan-it")
            done.set()

        task = await session.create_task(
            TaskSpec(
                title="plan",
                spec_type="plan",
                phases=[TaskPhase(id="plan-it")],
            ),
            owner="bob",
            blocking=False,
        )
        await asyncio.wait_for(done.wait(), timeout=1.0)

        updated = hub.peek_task(task.task_id)
        assert updated is not None
        phase = next(p for p in updated.spec.phases if p.id == "plan-it")
        assert phase.started_at is not None
        assert phase.completed_at is not None


# ---------------------------------------------------------------------------
# Session lifecycle interaction — task events bypass adapter 1Q1R
# ---------------------------------------------------------------------------


class TestSessionBypass:
    @pytest.mark.asyncio
    async def test_task_events_do_not_close_consulting_session(
        self, wired, session
    ) -> None:
        """Consulting sessions auto-close on the owner's first text reply.

        A task owner that emits many phase / progress events followed by a
        result should NOT trip that rule — task events are outside the
        adapter's delivery rules, so the session stays ACTIVE until an
        actual text reply (or explicit close) arrives.
        """

        hub, _store, _hc, _alice_client, bob_client, _alice, _bob = wired
        done = asyncio.Event()

        @bob_client.on_task("chat")
        async def _run(envelope, task: Task, client) -> None:  # noqa: ANN001
            await task.phase_entered("think")
            await task.progress(tokens=120)
            await task.phase_completed("think")
            await task.result({"message": "ok"})
            done.set()

        await session.create_task(
            TaskSpec(
                title="think-and-answer",
                spec_type="chat",
                phases=[TaskPhase(id="think")],
            ),
            owner="bob",
            blocking=False,
        )
        await asyncio.wait_for(done.wait(), timeout=1.0)
        await asyncio.sleep(0)

        refreshed = hub.peek_session(session.session_id)
        assert refreshed is not None
        assert refreshed.state is SessionState.ACTIVE

    @pytest.mark.asyncio
    async def test_multiple_tasks_in_one_consulting_session(
        self, wired, session
    ) -> None:
        hub, _store, _hc, _alice_client, bob_client, _alice, _bob = wired
        counters = {"done": 0}
        events = [asyncio.Event(), asyncio.Event()]

        @bob_client.on_task("counted")
        async def _run(envelope, task: Task, client) -> None:  # noqa: ANN001
            idx = counters["done"]
            counters["done"] += 1
            await task.result({"idx": idx})
            events[idx].set()

        t1 = await session.create_task(
            TaskSpec(title="t1", spec_type="counted"),
            owner="bob",
            blocking=False,
        )
        await asyncio.wait_for(events[0].wait(), timeout=1.0)
        t2 = await session.create_task(
            TaskSpec(title="t2", spec_type="counted"),
            owner="bob",
            blocking=False,
        )
        await asyncio.wait_for(events[1].wait(), timeout=1.0)
        await asyncio.sleep(0)

        assert hub.peek_task(t1.task_id).state is TaskState.COMPLETED
        assert hub.peek_task(t2.task_id).state is TaskState.COMPLETED
        assert hub.peek_task(t1.task_id).result == {"idx": 0}
        assert hub.peek_task(t2.task_id).result == {"idx": 1}
        assert hub.peek_session(session.session_id).state is SessionState.ACTIVE

    @pytest.mark.asyncio
    async def test_session_close_cancels_active_tasks(self, wired) -> None:
        hub, _store, _hc, alice_client, bob_client, _alice, _bob = wired

        started = asyncio.Event()
        release = asyncio.Event()

        @bob_client.on_task("slow")
        async def _run(envelope, task: Task, client) -> None:  # noqa: ANN001
            await task.phase_entered("long")
            started.set()
            try:
                await asyncio.wait_for(release.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                pass  # test cleanup

        sess = await alice_client.open(SessionType.CONSULTING, target="bob")
        task = await sess.create_task(
            TaskSpec(
                title="hold",
                spec_type="slow",
                phases=[TaskPhase(id="long")],
            ),
            owner="bob",
            blocking=False,
        )
        await asyncio.wait_for(started.wait(), timeout=1.0)
        assert hub.peek_task(task.task_id).state is TaskState.RUNNING

        await sess.close()
        await asyncio.sleep(0)
        refreshed = hub.peek_task(task.task_id)
        assert refreshed is not None
        assert refreshed.state is TaskState.CANCELLED
        assert refreshed.error in ("explicit", "session_closed") or isinstance(
            refreshed.error, str
        )
        release.set()


# ---------------------------------------------------------------------------
# Authority + state-machine guards
# ---------------------------------------------------------------------------


class TestAuthorityGuards:
    @pytest.mark.asyncio
    async def test_non_owner_cannot_emit_task_events(self, wired, session) -> None:
        hub, _store, _hc, alice_client, bob_client, _alice, _bob = wired
        blocking = asyncio.Event()

        @bob_client.on_task("polite")
        async def _run(envelope, task: Task, client) -> None:  # noqa: ANN001
            await blocking.wait()

        task = await session.create_task(
            TaskSpec(title="polite", spec_type="polite"),
            owner="bob",
            blocking=False,
        )

        # Alice is the requester but not the owner — attempting to post
        # a task event through her client must be rejected by the hub.
        from autogen.beta.network.errors import AccessDeniedError
        from autogen.beta.network import Envelope
        from autogen.beta.network.envelope import EV_TASK_PROGRESS

        forged = Envelope(
            session_id=session.session_id,
            sender_id=alice_client.actor_id,
            recipient_id=None,
            event_type=EV_TASK_PROGRESS,
            event_data={"update": {"attempt": 1}},
            task_id=task.task_id,
        )
        with pytest.raises(AccessDeniedError):
            await alice_client._send_envelope(forged)

        blocking.set()
        await asyncio.sleep(0)

    @pytest.mark.asyncio
    async def test_terminal_task_rejects_further_events(self, wired, session) -> None:
        hub, _store, _hc, _alice_client, bob_client, _alice, _bob = wired
        done = asyncio.Event()
        task_ref: dict[str, Task] = {}

        @bob_client.on_task("terminal")
        async def _run(envelope, task: Task, client) -> None:  # noqa: ANN001
            task_ref["task"] = task
            await task.result("first-and-only")
            done.set()

        task = await session.create_task(
            TaskSpec(title="t", spec_type="terminal"),
            owner="bob",
            blocking=False,
        )
        await asyncio.wait_for(done.wait(), timeout=1.0)
        await asyncio.sleep(0)
        assert hub.peek_task(task.task_id).state is TaskState.COMPLETED

        # A second result emission must raise TaskStateError because the
        # task is already terminal.
        from autogen.beta.network.errors import TaskStateError

        with pytest.raises(TaskStateError):
            await task_ref["task"].result("late-reply")

    @pytest.mark.asyncio
    async def test_hub_emitted_events_rejected_from_actors(
        self, wired, session
    ) -> None:
        _hub, _store, _hc, _alice_client, bob_client, _alice, _bob = wired
        block = asyncio.Event()

        @bob_client.on_task("hold")
        async def _run(envelope, task: Task, client) -> None:  # noqa: ANN001
            await block.wait()

        task = await session.create_task(
            TaskSpec(title="t", spec_type="hold"),
            owner="bob",
            blocking=False,
        )

        # Bob (the owner) must not be able to forge an ag2.task.cancelled
        # event — cancellation is a hub-only transition driven via
        # Hub.cancel_task.
        from autogen.beta.network.errors import TaskStateError
        from autogen.beta.network import Envelope
        from autogen.beta.network.envelope import EV_TASK_CANCELLED

        forged = Envelope(
            session_id=session.session_id,
            sender_id=bob_client.actor_id,
            event_type=EV_TASK_CANCELLED,
            event_data={"reason": "bored"},
            task_id=task.task_id,
        )
        with pytest.raises(TaskStateError):
            await bob_client._send_envelope(forged)

        block.set()
        await asyncio.sleep(0)
