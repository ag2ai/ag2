# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from contextlib import suppress

import pytest

from ag2.acp.sessions import (
    SessionBusyError,
    SessionConfig,
    SessionLimitError,
    SessionStore,
    UnknownSessionError,
)
from ag2.events import ModelMessage


@pytest.mark.asyncio
class TestIdentity:
    async def test_each_session_gets_a_distinct_id_and_stream(self) -> None:
        store = SessionStore()

        first = await store.create()
        second = await store.create()

        assert first.session_id != second.session_id
        assert first.stream_id != second.stream_id

    async def test_get_returns_the_same_session(self) -> None:
        store = SessionStore()

        created = await store.create()

        assert await store.get(created.session_id) is created

    async def test_unknown_id_raises(self) -> None:
        store = SessionStore()

        with pytest.raises(UnknownSessionError):
            await store.get("never-issued")

    async def test_client_context_is_captured_verbatim(self) -> None:
        store = SessionStore()

        session = await store.create(cwd="/work", additional_directories=["/extra"], meta={"ag2.space": {"room": "!r"}})

        assert session.cwd == "/work"
        assert session.additional_directories == ["/extra"]
        assert session.meta == {"ag2.space": {"room": "!r"}}


@pytest.mark.asyncio
class TestHistoryIsolation:
    async def test_two_sessions_never_share_history(self) -> None:
        store = SessionStore()
        first = await store.create()
        second = await store.create()

        await store.stream(first).history.replace([ModelMessage("only for first")])

        assert [e.content for e in await store.stream(first).history.get_events()] == ["only for first"]
        assert list(await store.stream(second).history.get_events()) == []

    async def test_history_accumulates_across_turns_of_one_session(self) -> None:
        store = SessionStore()
        session = await store.create()

        await store.stream(session).history.replace([ModelMessage("turn one")])
        # A *fresh* stream object per turn still reads the session's history back.
        later = store.stream(session)

        assert [e.content for e in await later.history.get_events()] == ["turn one"]

    async def test_a_session_keeps_one_stream_for_its_lifetime(self) -> None:
        """A stream carries an inbox and background tasks, not just history.

        Handing out a fresh object per turn would strand anything a background
        task delivered after its own turn finished.
        """
        store = SessionStore()
        session = await store.create()

        assert store.stream(session) is store.stream(session)

    async def test_a_message_enqueued_during_one_turn_survives_to_the_next(self) -> None:
        store = SessionStore()
        session = await store.create()

        store.stream(session).enqueue("late result")

        assert store.stream(session).pending_messages


@pytest.mark.asyncio
class TestQueueing:
    async def test_concurrent_prompts_on_one_session_run_one_at_a_time(self) -> None:
        store = SessionStore()
        session = await store.create()
        order: list[str] = []

        async def prompt(label: str) -> None:
            async with session.turn():
                order.append(f"start {label}")
                await asyncio.sleep(0.01)
                order.append(f"end {label}")

        await asyncio.gather(prompt("a"), prompt("b"))

        # Interleaving would give "start a", "start b", "end a", "end b".
        assert order in (
            ["start a", "end a", "start b", "end b"],
            ["start b", "end b", "start a", "end a"],
        )

    async def test_the_running_turn_does_not_count_against_the_queue(self) -> None:
        store = SessionStore.from_config(SessionConfig(max_queued=1))
        session = await store.create()
        release = asyncio.Event()

        async def blocked() -> None:
            async with session.turn():
                await release.wait()

        running = asyncio.create_task(blocked())
        await asyncio.sleep(0)  # let it take the lock

        assert session.queued == 0  # the holder is running, not queued

        queued = asyncio.create_task(blocked())
        await asyncio.sleep(0)

        assert session.queued == 1

        release.set()
        await asyncio.gather(running, queued)

    async def test_queue_overflow_is_rejected(self) -> None:
        store = SessionStore.from_config(SessionConfig(max_queued=2))
        session = await store.create()
        release = asyncio.Event()

        async def blocked() -> None:
            async with session.turn():
                await release.wait()

        running = asyncio.create_task(blocked())
        await asyncio.sleep(0)  # let it take the lock
        waiters = [asyncio.create_task(blocked()) for _ in range(2)]
        await asyncio.sleep(0)  # let both queue behind it

        with pytest.raises(SessionBusyError):
            async with session.turn():
                pass  # pragma: no cover - the guard fires before the body

        release.set()
        await asyncio.gather(running, *waiters)

    async def test_queue_drains_and_the_session_is_reusable(self) -> None:
        store = SessionStore.from_config(SessionConfig(max_queued=1))
        session = await store.create()

        async with session.turn():
            pass

        assert session.queued == 0
        async with session.turn():
            pass


@pytest.mark.asyncio
class TestCancellation:
    async def test_cancel_stops_the_running_turn(self) -> None:
        store = SessionStore()
        session = await store.create()
        started = asyncio.Event()

        async def never_finishes() -> None:
            async with session.turn():
                started.set()
                await asyncio.Event().wait()

        session.turn_task = asyncio.create_task(never_finishes())
        await started.wait()

        await session.cancel()

        with pytest.raises(asyncio.CancelledError):
            await session.turn_task

    async def test_cancel_also_drops_prompts_queued_behind(self) -> None:
        store = SessionStore()
        session = await store.create()
        started = asyncio.Event()
        reached_agent = False

        async def running() -> None:
            async with session.turn():
                started.set()
                await asyncio.Event().wait()

        async def queued() -> None:
            nonlocal reached_agent
            async with session.turn():
                reached_agent = True

        session.turn_task = asyncio.create_task(running())
        await started.wait()
        waiter = asyncio.create_task(queued())
        await asyncio.sleep(0)

        await session.cancel()

        with pytest.raises(asyncio.CancelledError):
            await session.turn_task
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert reached_agent is False

    async def test_cancel_leaves_other_sessions_untouched(self) -> None:
        store = SessionStore()
        cancelled = await store.create()
        other = await store.create()
        finished = False

        async def work() -> None:
            nonlocal finished
            async with other.turn():
                await asyncio.sleep(0.01)
                finished = True

        task = asyncio.create_task(work())
        await asyncio.sleep(0)

        await cancelled.cancel()
        await task

        assert finished is True

    async def test_events_already_emitted_survive_a_cancel(self) -> None:
        store = SessionStore()
        session = await store.create()
        await store.stream(session).history.replace([ModelMessage("already streamed")])

        await session.cancel()

        assert [e.content for e in await store.stream(session).history.get_events()] == ["already streamed"]


@pytest.mark.asyncio
class TestEviction:
    async def test_lru_overflow_drops_the_oldest_session(self) -> None:
        store = SessionStore(max_sessions=2)

        first = await store.create()
        await store.create()
        await store.create()

        assert len(store) == 2
        with pytest.raises(UnknownSessionError):
            await store.get(first.session_id)

    async def test_touching_a_session_saves_it_from_eviction(self) -> None:
        store = SessionStore(max_sessions=2)
        first = await store.create()
        second = await store.create()

        await store.get(first.session_id)  # first is now most-recently used
        await store.create()

        assert await store.get(first.session_id) is first
        with pytest.raises(UnknownSessionError):
            await store.get(second.session_id)

    async def test_idle_sessions_expire(self) -> None:
        now = 0.0
        store = SessionStore(ttl=10.0, clock=lambda: now)
        session = await store.create()

        now = 11.0

        with pytest.raises(UnknownSessionError):
            await store.get(session.session_id)

    async def test_evicted_history_is_deleted(self) -> None:
        store = SessionStore(max_sessions=1)
        first = await store.create()
        await store.stream(first).history.replace([ModelMessage("gone")])

        await store.create()  # evicts first

        assert list(await store.stream(first).history.get_events()) == []


@pytest.mark.asyncio
class TestClose:
    async def test_close_drops_the_session_and_its_history(self) -> None:
        store = SessionStore()
        session = await store.create()
        await store.stream(session).history.replace([ModelMessage("gone")])

        await store.close(session.session_id)

        with pytest.raises(UnknownSessionError):
            await store.get(session.session_id)
        assert list(await store.stream(session).history.get_events()) == []

    async def test_close_cancels_live_work_and_waits_for_it(self) -> None:
        """Unlike eviction, an explicit close *does* stop work in progress.

        It also waits for the turn to unwind before deleting history, so the
        agent's last writes cannot land in a store that has already been purged.
        """
        store = SessionStore()
        session = await store.create()
        started = asyncio.Event()

        async def running() -> None:
            async with session.turn():
                started.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(running())
        session.turn_task = task
        await started.wait()

        await store.close(session.session_id)

        assert task.done()
        assert task.cancelled()

    async def test_closing_an_unknown_session_raises(self) -> None:
        store = SessionStore()

        with pytest.raises(UnknownSessionError):
            await store.close("never-issued")

    async def test_aclose_drops_every_session(self) -> None:
        store = SessionStore()
        first = await store.create()
        second = await store.create()

        await store.aclose()

        assert len(store) == 0
        for session in (first, second):
            with pytest.raises(UnknownSessionError):
                await store.get(session.session_id)


def test_invalid_bounds_are_rejected() -> None:
    with pytest.raises(ValueError):
        SessionStore(max_sessions=0)
    with pytest.raises(ValueError):
        SessionStore(ttl=0)
    with pytest.raises(ValueError):
        SessionStore(max_queued=0)


@pytest.mark.asyncio
class TestEvictionSparesLiveWork:
    """An eviction policy is a memory bound, not a licence to kill running turns."""

    @staticmethod
    async def _busy(session: object) -> "asyncio.Task[None]":
        started = asyncio.Event()

        async def hold() -> None:
            async with session.turn():  # type: ignore[attr-defined]
                started.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(hold())
        session.turn_task = task  # type: ignore[attr-defined]
        await started.wait()
        return task

    async def test_pressure_on_the_cap_does_not_cancel_a_running_turn(self) -> None:
        """Admission fails instead — see :class:`TestTheCapIsAHardBound`."""
        store = SessionStore(max_sessions=1)
        busy = await store.create()
        task = await self._busy(busy)

        with pytest.raises(SessionLimitError):
            await store.create()
        await asyncio.sleep(0.01)

        assert not task.done()
        task.cancel()

    async def test_pressure_on_the_cap_does_not_delete_a_running_session_s_history(self) -> None:
        store = SessionStore(max_sessions=1)
        busy = await store.create()
        await store.stream(busy).history.replace([ModelMessage("mid-turn work")])
        task = await self._busy(busy)

        with pytest.raises(SessionLimitError):
            await store.create()
        await asyncio.sleep(0.01)

        assert [e.content for e in await store.stream(busy).history.get_events()] == ["mid-turn work"]
        task.cancel()

    async def test_an_idle_session_is_still_evicted(self) -> None:
        store = SessionStore(max_sessions=1)
        idle = await store.create()

        await store.create()

        with pytest.raises(UnknownSessionError):
            await store.get(idle.session_id)

    async def test_the_idle_victim_is_chosen_over_the_busy_one(self) -> None:
        store = SessionStore(max_sessions=2)
        busy = await store.create()
        idle = await store.create()
        task = await self._busy(busy)

        await store.create()

        assert await store.get(busy.session_id) is busy
        with pytest.raises(UnknownSessionError):
            await store.get(idle.session_id)
        task.cancel()

    async def test_ttl_does_not_expire_a_session_mid_turn(self) -> None:
        now = 0.0
        store = SessionStore(ttl=10.0, clock=lambda: now)
        busy = await store.create()
        task = await self._busy(busy)

        now = 11.0
        await store.create()  # any access runs the expiry sweep

        assert await store.get(busy.session_id) is busy
        assert not task.done()
        task.cancel()


@pytest.mark.asyncio
class TestEveryIssuedIdIsUsable:
    """A session id handed to a Client must still be there when it is used."""

    async def test_room_is_made_before_the_session_exists(self) -> None:
        """A session that gets created is never a candidate for its own eviction.

        Admission now runs *before* the session is built, so the id handed back
        cannot already have been evicted to make room for itself.
        """
        store = SessionStore(max_sessions=2)
        idle = await store.create()

        fresh = await store.create()

        assert await store.get(fresh.session_id) is fresh
        assert await store.get(idle.session_id) is idle

    async def test_the_cap_still_holds_when_an_idle_victim_exists(self) -> None:
        store = SessionStore(max_sessions=1)
        old = await store.create()

        fresh = await store.create()

        assert await store.get(fresh.session_id) is fresh
        with pytest.raises(UnknownSessionError):
            await store.get(old.session_id)


@pytest.mark.asyncio
class TestTtlMeasuresIdleTime:
    """Expiry is about being unused, not about when the last prompt started."""

    async def test_a_turn_longer_than_the_ttl_does_not_expire_its_session(self) -> None:
        now = 0.0
        store = SessionStore(ttl=10.0, clock=lambda: now)
        session = await store.create()
        await store.stream(session).history.replace([ModelMessage("slow work")])

        async with session.turn():
            now = 25.0  # the turn itself outlives the TTL

        assert await store.get(session.session_id) is session
        assert [e.content for e in await store.stream(session).history.get_events()] == ["slow work"]

    async def test_the_clock_restarts_when_a_turn_finishes(self) -> None:
        now = 0.0
        store = SessionStore(ttl=10.0, clock=lambda: now)
        session = await store.create()

        async with session.turn():
            now = 25.0

        now = 30.0  # only 5s idle since the turn ended
        assert await store.get(session.session_id) is session

        now = 45.0  # now genuinely idle past the TTL
        with pytest.raises(UnknownSessionError):
            await store.get(session.session_id)


@pytest.mark.asyncio
class TestTheCapIsAHardBound:
    """A cap that stretches while turns are in flight is not a cap at all."""

    @staticmethod
    async def _busy(store: SessionStore) -> "tuple[object, asyncio.Task[None]]":
        session = await store.create()
        started = asyncio.Event()

        async def hold() -> None:
            async with session.turn():  # type: ignore[attr-defined]
                started.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(hold())
        session.turn_task = task  # type: ignore[attr-defined]
        await started.wait()
        return session, task

    async def test_admission_is_refused_when_nothing_can_be_evicted(self) -> None:
        store = SessionStore(max_sessions=1)
        _busy, task = await self._busy(store)

        with pytest.raises(SessionLimitError):
            await store.create()

        assert len(store) == 1
        task.cancel()

    async def test_a_client_cannot_grow_the_registry_by_staying_busy(self) -> None:
        store = SessionStore(max_sessions=2)
        tasks = []
        for _ in range(2):
            _session, task = await self._busy(store)
            tasks.append(task)

        for _ in range(5):
            with pytest.raises(SessionLimitError):
                await store.create()

        assert len(store) == 2
        for task in tasks:
            task.cancel()

    async def test_a_slot_frees_up_once_a_turn_finishes(self) -> None:
        store = SessionStore(max_sessions=1)
        _session, task = await self._busy(store)

        with pytest.raises(SessionLimitError):
            await store.create()

        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

        fresh = await store.create()
        assert await store.get(fresh.session_id) is fresh
        assert len(store) == 1
