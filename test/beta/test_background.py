# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for autogen.beta.background — BackgroundTask and run_in_background."""

import asyncio

import pytest

from autogen.beta import Agent
from autogen.beta.background import BackgroundTask, run_in_background
from autogen.beta.testing import TestConfig

# ---------------------------------------------------------------------------
# BackgroundTask — unit tests using plain asyncio coroutines
# ---------------------------------------------------------------------------


class TestBackgroundTask:
    async def _make(self, coro) -> BackgroundTask:  # type: ignore[type-arg]
        task: asyncio.Task = asyncio.create_task(coro)
        return BackgroundTask(task)

    @pytest.mark.asyncio
    async def test_done_before_and_after(self) -> None:
        async def _noop() -> str:
            return "hi"

        bt = await self._make(_noop())
        assert not bt.done()
        await bt
        assert bt.done()

    @pytest.mark.asyncio
    async def test_cancelled_after_cancel(self) -> None:
        gate = asyncio.Event()

        async def _wait() -> None:
            await gate.wait()

        bt = await self._make(_wait())
        assert not bt.cancelled()
        bt.cancel()
        with pytest.raises(asyncio.CancelledError):
            await bt
        assert bt.cancelled()

    @pytest.mark.asyncio
    async def test_cancel_returns_false_when_done(self) -> None:
        async def _noop() -> str:
            return "done"

        bt = await self._make(_noop())
        await bt
        assert not bt.cancel()

    @pytest.mark.asyncio
    async def test_result_after_completion(self) -> None:
        async def _answer() -> int:
            return 42

        bt = await self._make(_answer())
        await bt
        assert bt.result() == 42

    @pytest.mark.asyncio
    async def test_result_raises_when_not_done(self) -> None:
        gate = asyncio.Event()

        async def _wait() -> None:
            await gate.wait()

        bt = await self._make(_wait())
        with pytest.raises(asyncio.InvalidStateError):
            bt.result()
        bt.cancel()
        with pytest.raises(asyncio.CancelledError):
            await bt

    @pytest.mark.asyncio
    async def test_exception_on_success_is_none(self) -> None:
        async def _ok() -> str:
            return "fine"

        bt = await self._make(_ok())
        await bt
        assert bt.exception() is None

    @pytest.mark.asyncio
    async def test_exception_propagates(self) -> None:
        async def _boom() -> None:
            raise ValueError("oops")

        bt = await self._make(_boom())
        with pytest.raises(ValueError, match="oops"):
            await bt
        exc = bt.exception()
        assert isinstance(exc, ValueError)

    @pytest.mark.asyncio
    async def test_exception_raises_when_not_done(self) -> None:
        gate = asyncio.Event()

        async def _wait() -> None:
            await gate.wait()

        bt = await self._make(_wait())
        with pytest.raises(asyncio.InvalidStateError):
            bt.exception()
        bt.cancel()
        with pytest.raises(asyncio.CancelledError):
            await bt

    @pytest.mark.asyncio
    async def test_await_returns_task_result(self) -> None:
        async def _val() -> str:
            return "value"

        bt = await self._make(_val())
        result = await bt
        assert result == "value"


# ---------------------------------------------------------------------------
# run_in_background — integration tests with TestConfig + Agent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRunInBackground:
    async def test_returns_background_task_immediately(self) -> None:
        config = TestConfig("summary complete")
        agent = Agent("analyst", config=config)

        task = run_in_background(agent, "Summarise.")
        assert isinstance(task, BackgroundTask)

    async def test_agent_reply_accessible_after_await(self) -> None:
        config = TestConfig("summary complete")
        agent = Agent("analyst", config=config)

        task = run_in_background(agent, "Summarise.")
        reply = await task
        text = await reply.content()
        assert text == "summary complete"

    async def test_not_done_before_yield(self) -> None:
        config = TestConfig("done")
        agent = Agent("worker", config=config)

        task = run_in_background(agent, "Go.")
        assert not task.done()
        await task
        assert task.done()

    async def test_two_tasks_run_concurrently(self) -> None:
        config_a = TestConfig("alpha")
        config_b = TestConfig("beta")
        agent_a = Agent("alpha", config=config_a)
        agent_b = Agent("beta", config=config_b)

        task_a = run_in_background(agent_a, "Task A.")
        task_b = run_in_background(agent_b, "Task B.")

        replies = await asyncio.gather(task_a, task_b)
        texts = [await r.content() for r in replies]
        assert "alpha" in texts
        assert "beta" in texts

    async def test_kwargs_forwarded_to_ask(self) -> None:
        from autogen.beta.stream import MemoryStream

        config = TestConfig("ok")
        agent = Agent("streamer", config=config)
        stream = MemoryStream()

        task = run_in_background(agent, "Hello.", stream=stream)
        await task

        history = await stream.history.storage.get_history(stream.id)
        assert len(history) > 0


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_all_exports() -> None:
    from autogen.beta import background

    assert "BackgroundTask" in background.__all__
    assert "run_in_background" in background.__all__
