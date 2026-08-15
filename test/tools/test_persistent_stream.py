# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import MagicMock

import pytest

from ag2 import Agent, Context, tool
from ag2.events import ModelMessage, ModelResponse, ToolCallEvent, ToolCallsEvent, Usage
from ag2.history import MemoryStorage
from ag2.stream import MemoryStream
from ag2.testing import TestConfig
from ag2.tools.subagents.persistent_stream import persistent_stream
from ag2.tools.subagents.run_task import run_task
from ag2.usage import UsageReport


@pytest.fixture()
def storage() -> MemoryStorage:
    return MemoryStorage()


@pytest.fixture()
def parent_stream(storage: MemoryStorage) -> MemoryStream:
    return MemoryStream(storage=storage)


@pytest.fixture()
def ctx(parent_stream: MemoryStream) -> Context:
    return Context(stream=parent_stream, dependencies={})


def _make_agent(name: str = "helper") -> MagicMock:
    agent = MagicMock()
    agent.name = name
    return agent


class TestPersistentStream:
    def test_returns_memory_stream(self, ctx: Context) -> None:
        factory = persistent_stream()
        agent = _make_agent()

        result = factory(agent, ctx)

        assert isinstance(result, MemoryStream)

    def test_reuses_same_stream_id_on_second_call(self, ctx: Context) -> None:
        factory = persistent_stream()
        agent = _make_agent()

        first = factory(agent, ctx)
        second = factory(agent, ctx)

        assert first.id == second.id

    def test_different_agents_get_different_streams(self, ctx: Context) -> None:
        factory = persistent_stream()
        agent_a = _make_agent("alice")
        agent_b = _make_agent("bob")

        stream_a = factory(agent_a, ctx)
        stream_b = factory(agent_b, ctx)

        assert stream_a.id != stream_b.id

    def test_stores_stream_id_in_dependencies(self, ctx: Context) -> None:
        factory = persistent_stream()
        agent = _make_agent("helper")

        stream = factory(agent, ctx)

        assert ctx.dependencies["ag:helper:stream"] == stream.id

    def test_uses_parent_storage_backend(self, ctx: Context, storage: MemoryStorage) -> None:
        factory = persistent_stream()
        agent = _make_agent()

        stream = factory(agent, ctx)

        assert stream.history.storage is storage

    def test_independent_contexts_get_independent_streams(self, storage: MemoryStorage) -> None:
        factory = persistent_stream()
        agent = _make_agent()

        ctx1 = Context(stream=MemoryStream(storage=storage), dependencies={})
        ctx2 = Context(stream=MemoryStream(storage=storage), dependencies={})

        stream1 = factory(agent, ctx1)
        stream2 = factory(agent, ctx2)

        assert stream1.id != stream2.id


@pytest.mark.asyncio
class TestUsageAccountingOnAReusedStream:
    """Delegating repeatedly to the same long-lived worker must add up linearly.

    The rollup the parent receives carries *this invocation's* spend; the
    ``TaskResult.usage`` field keeps the cumulative reading of the worker's
    stream. On a fresh stream per call the two coincide — reusing the stream is
    what tells them apart.
    """

    async def test_repeated_delegations_do_not_inflate_the_parent_total(self, ctx: Context) -> None:
        billed = Usage(prompt_tokens=100, completion_tokens=10, total_tokens=110)
        worker = Agent(
            "worker",
            config=TestConfig(*(ModelResponse(ModelMessage("done"), usage=billed) for _ in range(3))),
        )
        factory = persistent_stream()

        readings = [
            (await run_task(worker, "go", parent_context=ctx, stream=factory(worker, ctx))).usage for _ in range(3)
        ]

        # The field is the worker's running total across the session.
        assert readings == [billed, billed + billed, billed + billed + billed]

        # The parent's report equals what was actually spent — not the sum of
        # three cumulative readings.
        spent = billed + billed + billed
        report = UsageReport.from_events(await ctx.stream.history.get_events())
        assert report.total == spent
        assert report.by_kind == {"subtask": spent}

    async def test_a_failed_delegation_on_a_reused_stream_reports_only_its_own_spend(self, ctx: Context) -> None:
        """The failure path must not carry the over-count either.

        The worker's downstream API works once and then breaks, so the second
        delegation dies after billing. The parent must be told what that attempt
        spent, not everything the worker has ever spent.
        """
        calls = 0

        @tool
        def flaky() -> str:
            """A downstream API that works once and then breaks."""
            nonlocal calls
            calls += 1
            if calls > 1:
                raise RuntimeError("downstream API is down")
            return "ok"

        dispatch = Usage(prompt_tokens=100, completion_tokens=10, total_tokens=110)
        wrapup = Usage(prompt_tokens=40, completion_tokens=4, total_tokens=44)
        worker = Agent(
            "worker",
            config=TestConfig(
                ModelResponse(
                    tool_calls=ToolCallsEvent(calls=[ToolCallEvent(name="flaky", arguments="{}")]),
                    usage=dispatch,
                ),
                ModelResponse(ModelMessage("done"), usage=wrapup),
            ),
            tools=[flaky],
        )
        factory = persistent_stream()

        ok = await run_task(worker, "go", parent_context=ctx, stream=factory(worker, ctx))
        failed = await run_task(worker, "go again", parent_context=ctx, stream=factory(worker, ctx))

        assert ok.completed is True
        assert failed.completed is False
        # Cumulative on the field: everything this worker has spent on its stream.
        assert failed.usage == dispatch + wrapup + dispatch

        # Per-invocation on the parent: the successful attempt's two calls, then
        # the failed attempt's single call.
        report = UsageReport.from_events(await ctx.stream.history.get_events())
        assert report.total == dispatch + wrapup + dispatch
        assert report.by_kind == {"subtask": dispatch + wrapup + dispatch}


@pytest.mark.asyncio
class TestConcurrentDelegationsToOnePersistentWorker:
    """Two sibling delegations into one persistent worker — the ``asyncio.gather``
    shape ``run_task``'s docstring anticipates.

    They run one after another rather than interleaving their writes into the
    history they share, and the parent is billed for what the worker really
    spent.
    """

    async def test_concurrent_delegations_do_not_interleave(self, ctx: Context) -> None:
        active = 0
        peak = 0

        @tool
        async def slow_step() -> str:
            """Holds long enough for a racing delegation to arrive."""
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(0.05)
            finally:
                active -= 1
            return "ok"

        call_the_tool = ModelResponse(
            tool_calls=ToolCallsEvent(calls=[ToolCallEvent(name="slow_step", arguments="{}")])
        )
        wrap_up = ModelResponse(ModelMessage("done"))
        worker = Agent(
            "worker",
            config=TestConfig(call_the_tool, wrap_up, call_the_tool, wrap_up),
            tools=[slow_step],
        )
        factory = persistent_stream()

        await asyncio.gather(
            run_task(worker, "first", parent_context=ctx, stream=factory(worker, ctx)),
            run_task(worker, "second", parent_context=ctx, stream=factory(worker, ctx)),
        )

        assert peak == 1, "delegations into one persistent worker must not overlap"

    async def test_concurrent_delegations_do_not_inflate_the_parent_total(self, ctx: Context) -> None:
        billed = Usage(prompt_tokens=100, completion_tokens=10, total_tokens=110)
        worker = Agent(
            "worker",
            config=TestConfig(*(ModelResponse(ModelMessage("done"), usage=billed) for _ in range(2))),
        )
        factory = persistent_stream()

        first, second = await asyncio.gather(
            run_task(worker, "first", parent_context=ctx, stream=factory(worker, ctx)),
            run_task(worker, "second", parent_context=ctx, stream=factory(worker, ctx)),
        )

        # The parent is billed once per delegation — not once per delegation
        # times every sibling whose events its accumulator could see.
        spent = billed + billed
        report = UsageReport.from_events(await ctx.stream.history.get_events())
        assert report.total == spent
        assert report.by_kind == {"subtask": spent}

        # The field keeps its documented meaning: the worker's running total on
        # its stream. Whichever delegation went second reads both.
        readings = sorted([first.usage, second.usage], key=lambda usage: usage.total_tokens)
        assert readings == [billed, spent]
