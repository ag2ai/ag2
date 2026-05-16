# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for AgentScheduler."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from autogen.beta import Agent, AgentScheduler, ScheduledJob
from autogen.beta.scheduler import _resolve_prompt
from autogen.beta.testing import TestConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agent(response: str = "ok") -> Agent:
    return Agent("test", config=TestConfig(response))


# ---------------------------------------------------------------------------
# _resolve_prompt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_resolve_prompt_string() -> None:
    assert await _resolve_prompt("hello") == "hello"


@pytest.mark.asyncio()
async def test_resolve_prompt_sync_callable() -> None:
    assert await _resolve_prompt(lambda: "dynamic") == "dynamic"


@pytest.mark.asyncio()
async def test_resolve_prompt_async_callable() -> None:
    async def factory() -> str:
        return "async-prompt"

    assert await _resolve_prompt(factory) == "async-prompt"


# ---------------------------------------------------------------------------
# ScheduledJob
# ---------------------------------------------------------------------------


def test_scheduled_job_name_defaults_to_watch_class() -> None:
    from autogen.beta.watch import IntervalWatch

    job = ScheduledJob(IntervalWatch(60), _agent(), "ping")
    assert "IntervalWatch" in job.name


def test_scheduled_job_custom_name() -> None:
    from autogen.beta.watch import IntervalWatch

    job = ScheduledJob(IntervalWatch(60), _agent(), "ping", name="my-job")
    assert job.name == "my-job"


def test_scheduled_job_not_armed_initially() -> None:
    from autogen.beta.watch import IntervalWatch

    job = ScheduledJob(IntervalWatch(60), _agent(), "ping")
    assert not job.is_armed


# ---------------------------------------------------------------------------
# AgentScheduler registration
# ---------------------------------------------------------------------------


def test_interval_returns_scheduled_job() -> None:
    sched = AgentScheduler()
    job = sched.interval(60, agent=_agent(), prompt="ping")
    assert isinstance(job, ScheduledJob)
    assert "60" in job.name


def test_cron_returns_scheduled_job() -> None:
    sched = AgentScheduler()
    job = sched.cron("0 9 * * *", agent=_agent(), prompt="daily")
    assert isinstance(job, ScheduledJob)
    assert "0 9 * * *" in job.name


def test_delay_returns_scheduled_job() -> None:
    sched = AgentScheduler()
    job = sched.delay(5, agent=_agent(), prompt="once")
    assert isinstance(job, ScheduledJob)


def test_add_with_custom_watch() -> None:
    from autogen.beta.watch import CronWatch

    sched = AgentScheduler()
    watch = CronWatch("*/5 * * * *")
    job = sched.add(watch, agent=_agent(), prompt="custom", name="my-watch")
    assert job.name == "my-watch"


# ---------------------------------------------------------------------------
# Lifecycle — start / stop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_start_arms_registered_jobs() -> None:
    sched = AgentScheduler()
    job = sched.interval(999, agent=_agent(), prompt="ping")
    assert not job.is_armed

    await sched.start()
    assert job.is_armed

    await sched.stop()
    assert not job.is_armed


@pytest.mark.asyncio()
async def test_start_is_idempotent() -> None:
    sched = AgentScheduler()
    sched.interval(999, agent=_agent(), prompt="ping")

    await sched.start()
    await sched.start()  # second call is a no-op

    await sched.stop()


@pytest.mark.asyncio()
async def test_job_added_after_start_is_armed_immediately() -> None:
    sched = AgentScheduler()
    await sched.start()

    job = sched.interval(999, agent=_agent(), prompt="ping")
    assert job.is_armed

    await sched.stop()


@pytest.mark.asyncio()
async def test_context_manager_arms_and_stops() -> None:
    sched = AgentScheduler()
    job = sched.interval(999, agent=_agent(), prompt="ping")

    async with sched:
        assert job.is_armed

    assert not job.is_armed


# ---------------------------------------------------------------------------
# Job cancellation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_cancel_individual_job() -> None:
    sched = AgentScheduler()
    job_a = sched.interval(999, agent=_agent(), prompt="a")
    job_b = sched.interval(999, agent=_agent(), prompt="b")

    await sched.start()
    assert job_a.is_armed
    assert job_b.is_armed

    job_a.cancel()
    assert not job_a.is_armed
    assert job_b.is_armed  # b still running

    await sched.stop()


# ---------------------------------------------------------------------------
# Invocation — interval fires agent.ask()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_interval_fires_agent_ask() -> None:
    fired = asyncio.Event()
    on_reply = AsyncMock(side_effect=lambda _: fired.set())

    sched = AgentScheduler()
    sched.interval(0.05, agent=_agent("pong"), prompt="ping", on_reply=on_reply)

    async with sched:
        await asyncio.wait_for(fired.wait(), timeout=2.0)

    on_reply.assert_called_once()
    reply = on_reply.call_args[0][0]
    assert reply.body == "pong"


@pytest.mark.asyncio()
async def test_delay_fires_once() -> None:
    calls: list[str] = []

    async def on_reply(reply: object) -> None:
        calls.append("fired")

    sched = AgentScheduler()
    sched.delay(0.05, agent=_agent("done"), prompt="go", on_reply=on_reply)

    async with sched:
        await asyncio.sleep(0.3)

    # DelayWatch fires exactly once
    assert calls == ["fired"]


@pytest.mark.asyncio()
async def test_interval_prompt_factory_called_each_time() -> None:
    counter = MagicMock(return_value="msg")
    fired = asyncio.Event()
    call_count = 0

    async def on_reply(reply: object) -> None:
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            fired.set()

    sched = AgentScheduler()
    sched.interval(0.05, agent=_agent(), prompt=counter, on_reply=on_reply)

    async with sched:
        await asyncio.wait_for(fired.wait(), timeout=2.0)

    assert counter.call_count >= 2


@pytest.mark.asyncio()
async def test_agent_exception_does_not_stop_scheduler() -> None:
    """A failing agent.ask() is logged and the next interval still fires."""

    fire_count = 0
    fired_twice = asyncio.Event()

    async def on_reply(reply: object) -> None:
        nonlocal fire_count
        fire_count += 1
        if fire_count >= 2:
            fired_twice.set()

    bad_agent = _agent("ok")

    original_ask = bad_agent.ask
    call_count = 0

    async def flaky_ask(*args: object, **kwargs: object) -> object:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("transient error")
        return await original_ask(*args, **kwargs)

    bad_agent.ask = flaky_ask  # type: ignore[method-assign]

    sched = AgentScheduler()
    sched.interval(0.05, agent=bad_agent, prompt="ping", on_reply=on_reply)

    async with sched:
        await asyncio.wait_for(fired_twice.wait(), timeout=3.0)

    assert fire_count >= 2
