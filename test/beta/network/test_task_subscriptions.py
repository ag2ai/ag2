# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Task subscription fan-out — Phase 4.

Non-participant observers with a permissive ``rule.access.subscribe.sessions``
policy should see every ``ag2.task.*`` envelope in a session they subscribe
to. Participant observers (the requester) also see them — that's how the
blocking wait path works — but the explicit non-participant test pins the
contract that task subscriptions are no different from any other session
subscription.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
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
    SubscribeAccess,
    Task,
    TaskSpec,
    TaskState,
)
from autogen.beta.network.envelope import (
    EV_TASK_ASSIGNED,
    EV_TASK_PHASE_ENTERED,
    EV_TASK_PROGRESS,
    EV_TASK_RESULT,
)
from autogen.beta.network.rule import SUBSCRIBE_HUB_PUBLIC


@dataclass
class FakeReply:
    content: str


class FakeActor:
    def __init__(self, name: str) -> None:
        self.name = name

    async def ask(self, content: str, **kwargs: Any) -> FakeReply:
        return FakeReply(content=f"{self.name}: {content}")


def _public_rule() -> Rule:
    """A rule whose owner allows hub-public session observation.

    Every participant must vote for non-restrictive access for the
    hub's ``_can_observe_session`` check to pass.
    """

    return Rule(
        access=AccessBlock(
            subscribe=SubscribeAccess(sessions=SUBSCRIBE_HUB_PUBLIC),
        ),
        limits=LimitsBlock(),
    )


@pytest_asyncio.fixture
async def wired():
    store = MemoryKnowledgeStore()
    hub = Hub(store)
    link = LocalLink()
    link.on_connection(hub.connection_handler)
    hub_client = HubClient(hub, link)

    alice = FakeActor(name="alice")
    bob = FakeActor(name="bob")
    observer = FakeActor(name="observer")
    alice_client = await hub_client.register(
        alice, identity=ActorIdentity(name="alice"), rule=_public_rule()
    )
    bob_client = await hub_client.register(
        bob, identity=ActorIdentity(name="bob"), rule=_public_rule()
    )
    observer_client = await hub_client.register(
        observer,
        identity=ActorIdentity(name="observer"),
        rule=_public_rule(),
    )
    try:
        yield hub, hub_client, alice_client, bob_client, observer_client
    finally:
        await hub_client.close()
        await link.close()


async def _drain_task_events(
    queue: asyncio.Queue,
    task_id: str,
    *,
    expect_terminal: bool = True,
    max_wait: float = 1.0,
) -> list[str]:
    """Pull task-event types off a subscription queue until terminal or timeout."""

    from autogen.beta.network.envelope import TASK_TERMINAL_EVENT_TYPES

    seen: list[str] = []
    deadline = asyncio.get_event_loop().time() + max_wait
    while True:
        remaining = max(0.0, deadline - asyncio.get_event_loop().time())
        try:
            envelope = await asyncio.wait_for(queue.get(), timeout=remaining)
        except asyncio.TimeoutError:
            break
        if envelope.task_id != task_id:
            continue
        seen.append(envelope.event_type)
        if expect_terminal and envelope.event_type in TASK_TERMINAL_EVENT_TYPES:
            return seen
    return seen


class TestTaskSubscriptions:
    @pytest.mark.asyncio
    async def test_participant_sees_every_task_event(self, wired) -> None:
        _hub, _hc, alice_client, bob_client, _observer_client = wired

        @bob_client.on_task("phased")
        async def _run(envelope, task: Task, client) -> None:  # noqa: ANN001
            await task.phase_entered("collect")
            await task.progress(docs=2)
            await task.result({"answer": 42})

        sess = await alice_client.open(SessionType.CONSULTING, target="bob")

        # Alice subscribes before creating the task so she captures the
        # assignment envelope from the start.
        queue = await alice_client._open_subscription(
            session_id=sess.session_id, since=0
        )
        try:
            task = await sess.create_task(
                TaskSpec(title="t", spec_type="phased"),
                owner="bob",
                blocking=False,
            )
            types = await _drain_task_events(queue, task.task_id)
            assert EV_TASK_ASSIGNED in types
            assert EV_TASK_PHASE_ENTERED in types
            assert EV_TASK_PROGRESS in types
            assert EV_TASK_RESULT in types
        finally:
            await alice_client._close_subscription(queue)

    @pytest.mark.asyncio
    async def test_non_participant_observer_sees_task_events(self, wired) -> None:
        _hub, _hc, alice_client, bob_client, observer_client = wired

        @bob_client.on_task("watched")
        async def _run(envelope, task: Task, client) -> None:  # noqa: ANN001
            await task.phase_entered("work")
            await task.result("ok")

        sess = await alice_client.open(SessionType.CONSULTING, target="bob")

        # Observer is not a participant — subscribe requires the
        # session's participants AND the observer to allow
        # hub-public observation.
        queue = await observer_client._open_subscription(
            session_id=sess.session_id, since=0
        )
        try:
            task = await sess.create_task(
                TaskSpec(title="t", spec_type="watched"),
                owner="bob",
                blocking=False,
            )
            types = await _drain_task_events(queue, task.task_id)
            assert EV_TASK_ASSIGNED in types
            assert EV_TASK_PHASE_ENTERED in types
            assert EV_TASK_RESULT in types
        finally:
            await observer_client._close_subscription(queue)

    @pytest.mark.asyncio
    async def test_multiple_subscribers_all_see_the_task(self, wired) -> None:
        _hub, _hc, alice_client, bob_client, observer_client = wired

        @bob_client.on_task("fanout")
        async def _run(envelope, task: Task, client) -> None:  # noqa: ANN001
            await task.result("broadcasted")

        sess = await alice_client.open(SessionType.CONSULTING, target="bob")
        alice_q = await alice_client._open_subscription(
            session_id=sess.session_id, since=0
        )
        obs_q = await observer_client._open_subscription(
            session_id=sess.session_id, since=0
        )
        try:
            task = await sess.create_task(
                TaskSpec(title="t", spec_type="fanout"),
                owner="bob",
                blocking=False,
            )
            alice_types = await _drain_task_events(alice_q, task.task_id)
            obs_types = await _drain_task_events(obs_q, task.task_id)
            assert EV_TASK_RESULT in alice_types
            assert EV_TASK_RESULT in obs_types
        finally:
            await alice_client._close_subscription(alice_q)
            await observer_client._close_subscription(obs_q)

    @pytest.mark.asyncio
    async def test_subscription_shows_cancellation(self, wired) -> None:
        hub, _hc, alice_client, bob_client, observer_client = wired
        release = asyncio.Event()

        @bob_client.on_task("hold")
        async def _run(envelope, task: Task, client) -> None:  # noqa: ANN001
            await task.phase_entered("wait")
            await release.wait()

        sess = await alice_client.open(SessionType.CONSULTING, target="bob")
        queue = await observer_client._open_subscription(
            session_id=sess.session_id, since=0
        )
        try:
            task = await sess.create_task(
                TaskSpec(title="t", spec_type="hold"),
                owner="bob",
                blocking=False,
            )
            await hub.cancel_task(
                task.task_id, requested_by=alice_client.actor_id, reason="stop"
            )
            types = await _drain_task_events(queue, task.task_id)
            from autogen.beta.network.envelope import EV_TASK_CANCELLED

            assert EV_TASK_CANCELLED in types
        finally:
            await observer_client._close_subscription(queue)
            release.set()
