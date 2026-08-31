# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import pickle
import time
from collections.abc import Awaitable, Callable
from unittest.mock import MagicMock

import pytest

from ag2 import Agent, Context, TaskConfig
from ag2.events import (
    HumanInputRequest,
    HumanMessage,
    TaskFailed,
    ToolCallEvent,
    ToolCallsEvent,
    ToolResultsEvent,
)
from ag2.exceptions import (
    HumanInputError,
    HumanInputFailedError,
    HumanInputNotProvidedError,
    HumanInputTimeoutError,
)
from ag2.history import HUMAN_INPUT_ABANDONED_TOOL_RESULT
from ag2.middleware import approval_required
from ag2.stream import MemoryStream
from ag2.testing import TestConfig
from ag2.tools import tool
from ag2.tools.subagents import subagent_tool


@pytest.fixture()
def test_config() -> TestConfig:
    return TestConfig(
        ToolCallEvent(name="my_tool"),
        "result",
    )


@pytest.mark.asyncio()
async def test_sync_hitl(
    mock: MagicMock,
    test_config: TestConfig,
) -> None:
    async def my_tool(ctx: Context) -> str:
        mock(await ctx.input("Say smth", timeout=1.0))
        return ""

    def hitl_hook(event: HumanInputRequest) -> HumanMessage:
        mock.hitl(event.content)
        return HumanMessage("answer")

    agent = Agent(
        "",
        config=test_config,
        tools=[my_tool],
        hitl_hook=hitl_hook,
    )

    await agent.ask("Hi!")

    mock.assert_called_once_with("answer")
    mock.hitl.assert_called_once_with("Say smth")


@pytest.mark.asyncio()
async def test_async_hitl(
    mock: MagicMock,
    test_config: TestConfig,
) -> None:
    async def my_tool(ctx: Context) -> str:
        mock(await ctx.input("Say smth", timeout=1.0))
        return ""

    async def hitl_hook(event: HumanInputRequest) -> HumanMessage:
        return HumanMessage("answer")

    agent = Agent(
        "",
        config=test_config,
        tools=[my_tool],
        hitl_hook=hitl_hook,
    )

    await agent.ask("Hi!")

    mock.assert_called_once_with("answer")


@pytest.mark.asyncio()
async def test_hitl_decorator(
    mock: MagicMock,
    test_config: TestConfig,
) -> None:
    async def my_tool(ctx: Context) -> str:
        mock(await ctx.input("Say smth", timeout=1.0))
        return ""

    agent = Agent(
        "",
        config=test_config,
        tools=[my_tool],
    )

    @agent.hitl_hook
    def hitl_hook(event: HumanInputRequest) -> HumanMessage:
        return HumanMessage("answer")

    await agent.ask("Hi!")

    mock.assert_called_once_with("answer")


@pytest.mark.asyncio()
async def test_hitl_decorator_override(
    mock: MagicMock,
    test_config: TestConfig,
) -> None:
    async def my_tool(ctx: Context) -> str:
        mock(await ctx.input("Say smth", timeout=1.0))
        return ""

    agent = Agent(
        "",
        config=test_config,
        tools=[my_tool],
    )

    @agent.hitl_hook
    def overridden_hook(event: HumanInputRequest) -> HumanMessage:
        return HumanMessage("wrong")

    with pytest.warns(RuntimeWarning):

        @agent.hitl_hook
        def hitl_hook(event: HumanInputRequest) -> HumanMessage:
            return HumanMessage("answer")

    await agent.ask("Hi!")

    mock.assert_called_once_with("answer")


@pytest.mark.asyncio()
async def test_hitl_not_set(
    mock: MagicMock,
    test_config: TestConfig,
) -> None:
    async def my_tool(ctx: Context) -> str:
        try:
            await ctx.input("Say smth", timeout=1.0)
        except HumanInputNotProvidedError:
            mock()
        return ""

    agent = Agent(
        "",
        config=test_config,
        tools=[my_tool],
    )

    await agent.ask("Hi!")

    mock.assert_called_once()


# Timeouts the tests below rely on. Small enough that the suite pays
# milliseconds for them, and far enough apart that a slow CI box does not
# flip which one wins.
IMPATIENT = 0.05
SLOWER_THAN_TIMEOUT = 1.0


class ApprovalQueueDownError(RuntimeError):
    """Stands in for the application's own machinery failing."""


def _lenient(tool_name: str, arguments: str = "{}") -> TestConfig:
    """One tool call, then an answer, from a double that does not re-raise.

    ``TestConfig`` re-raises any ``ToolErrorEvent`` in the history by default, so
    a turn appears to fail under it whenever anything on the tool path raises —
    which hides whether *this* failure ends the turn or is quietly recorded as a
    tool result and answered around.
    """
    return TestConfig(
        ToolCallEvent(name=tool_name, arguments=arguments),
        "done",
        raise_tool_errors=False,
    )


@pytest.mark.asyncio()
class TestChannelFailureEndsTheTurn:
    """A question that never reached a human is not a tool that failed.

    Tool execution records a raising tool as a ``ToolErrorEvent`` and lets the
    turn carry on. Doing that to a human-input failure hands the model a
    traceback where an answer should be, and the caller a turn that reports
    success — so the model is free to route around an approval nobody was
    actually asked for.
    """

    async def test_a_missing_hook_ends_the_turn(self) -> None:
        executed = MagicMock()

        async def my_tool(ctx: Context) -> str:
            await ctx.input("Say smth", timeout=1.0)
            executed()
            return ""

        agent = Agent("", config=_lenient("my_tool"), tools=[my_tool])

        with pytest.raises(HumanInputNotProvidedError):
            await agent.ask("Hi!")

        executed.assert_not_called()

    async def test_a_failing_hook_ends_the_turn_carrying_its_cause(self) -> None:
        """One type out, the hook's own exception kept on ``cause``.

        Callers branch on what broke without every catch site downstream having
        to recognise an exception type it has never heard of.
        """

        async def my_tool(ctx: Context) -> str:
            return await ctx.input("Say smth", timeout=1.0)

        def hitl_hook(event: HumanInputRequest) -> HumanMessage:
            raise ApprovalQueueDownError("approval queue unreachable")

        agent = Agent("", config=_lenient("my_tool"), tools=[my_tool], hitl_hook=hitl_hook)

        with pytest.raises(HumanInputFailedError) as caught:
            await agent.ask("Hi!")

        assert isinstance(caught.value.cause, ApprovalQueueDownError)
        assert isinstance(caught.value.__cause__, ApprovalQueueDownError)

    async def test_nobody_answering_in_time_ends_the_turn(self) -> None:
        """A hook that never answers is nobody answering, not a slow tool.

        The hook runs inline inside the send, so a timeout that only covers the
        waiting never fires at all — and a hook that hangs hangs the turn.
        """
        executed = MagicMock()

        async def my_tool(ctx: Context) -> str:
            await ctx.input("Say smth", timeout=IMPATIENT)
            executed()
            return ""

        async def hitl_hook(event: HumanInputRequest) -> HumanMessage:
            # Longer than the timeout, short enough that a regression costs a
            # second rather than hanging CI.
            await asyncio.sleep(SLOWER_THAN_TIMEOUT)
            return HumanMessage("too late")

        agent = Agent("", config=_lenient("my_tool"), tools=[my_tool], hitl_hook=hitl_hook)

        with pytest.raises(HumanInputTimeoutError):
            await agent.ask("Hi!")

        executed.assert_not_called()

    async def test_a_hook_raising_its_own_timeout_is_not_read_as_nobody_answering(self) -> None:
        """Two different failures that happen to share an exception type.

        A hook whose own call timed out did reach the channel and got an error
        back; nobody answering is the deadline expiring on this side. Telling
        them apart by type alone cannot work, so the classification happens
        where the difference is still visible.
        """

        async def my_tool(ctx: Context) -> str:
            return await ctx.input("Say smth", timeout=SLOWER_THAN_TIMEOUT)

        def hitl_hook(event: HumanInputRequest) -> HumanMessage:
            raise TimeoutError("our own upstream call timed out")

        agent = Agent("", config=_lenient("my_tool"), tools=[my_tool], hitl_hook=hitl_hook)

        with pytest.raises(HumanInputFailedError) as caught:
            await agent.ask("Hi!")

        assert isinstance(caught.value.cause, TimeoutError)

    async def test_a_late_answer_does_not_release_a_gated_tool(self) -> None:
        """``approval_required``'s timeout is a control, so it has to bind."""
        executed = MagicMock()

        async def my_tool() -> str:
            executed()
            return ""

        async def hitl_hook(event: HumanInputRequest) -> HumanMessage:
            await asyncio.sleep(SLOWER_THAN_TIMEOUT)
            return HumanMessage("y")

        agent = Agent(
            "",
            config=_lenient("my_tool"),
            tools=[tool(my_tool, middleware=[approval_required(timeout=IMPATIENT)])],
            hitl_hook=hitl_hook,
        )

        with pytest.raises(HumanInputTimeoutError):
            await agent.ask("Hi!")

        executed.assert_not_called()

    async def test_the_failure_reaches_a_middleware_asking_on_a_tools_behalf(self) -> None:
        """``approval_required`` asks from around the tool, not from inside it.

        A different catch site in the executor, and the one that matters most:
        an approval that could not be requested must not read as the tool
        failing, or the model is invited to try another way.
        """
        executed = MagicMock()

        async def my_tool() -> str:
            executed()
            return ""

        agent = Agent(
            "",
            config=_lenient("my_tool"),
            tools=[tool(my_tool, middleware=[approval_required()])],
        )

        with pytest.raises(HumanInputNotProvidedError):
            await agent.ask("Hi!")

        executed.assert_not_called()

    async def test_a_tool_reraising_as_its_own_error_still_ends_the_turn(self) -> None:
        """The signal is a type, so nothing has to remember to preserve a tag.

        A tag hung on whatever the hook threw is gone the moment anything in
        between wraps it; a subclass of ``HumanInputError`` raised deliberately
        is a decision, not an accident.
        """

        async def my_tool(ctx: Context) -> str:
            try:
                return await ctx.input("Say smth", timeout=1.0)
            except HumanInputNotProvidedError as exc:
                raise HumanInputFailedError(exc) from exc

        agent = Agent("", config=_lenient("my_tool"), tools=[my_tool])

        with pytest.raises(HumanInputFailedError):
            await agent.ask("Hi!")

    async def test_a_tool_that_handles_missing_input_still_decides_for_itself(self) -> None:
        """Propagating is the default, not a veto: catching it still works."""
        handled = MagicMock()

        async def my_tool(ctx: Context) -> str:
            try:
                await ctx.input("Say smth", timeout=1.0)
            except HumanInputNotProvidedError:
                handled()
            return "carried on"

        agent = Agent("", config=_lenient("my_tool"), tools=[my_tool])

        await agent.ask("Hi!")

        handled.assert_called_once()


def _asking_tool() -> "tuple[Callable[[Context], Awaitable[str]], MagicMock]":
    """A tool that asks a human, and a spy on whether it ever got past the asking."""
    executed = MagicMock()

    async def ask_human(ctx: Context) -> str:
        answer = await ctx.input("Say smth", timeout=1.0)
        executed()
        return answer

    return ask_human, executed


@pytest.mark.asyncio()
async def test_a_delegated_question_nobody_can_answer_ends_the_parent_turn() -> None:
    """A sub-agent's unanswerable question is the same failure one level down.

    Delegation turns a child's exception into the delegating tool's output, so
    without this the parent's model reads "the subtask failed" and is invited to
    find another way to the same effect.
    """
    ask_human, executed = _asking_tool()
    child = Agent("child", config=_lenient("ask_human"), tools=[ask_human])
    parent = Agent(
        "parent",
        config=_lenient("delegate", '{"objective": "ask the human"}'),
        tools=[subagent_tool(child, name="delegate", description="delegate to the child")],
    )

    with pytest.raises(HumanInputNotProvidedError):
        await parent.ask("Hi!")

    executed.assert_not_called()


@pytest.mark.asyncio()
@pytest.mark.parametrize("parallel", [True, False], ids=["parallel", "sequential"])
async def test_an_unanswerable_subtask_ends_the_parent_turn(parallel: bool) -> None:
    """``run_subtasks`` fans out; the fan-out must not swallow the failure.

    ``subagent_tool`` and ``run_subtasks`` reach ``run_task`` by different
    routes: the parallel branch collects exceptions with
    ``return_exceptions=True`` and the sequential one catches them per task, and
    both used to render whatever came back as text. Either way the parent's
    model would read an unaskable question as a subtask that merely failed.
    """
    ask_human, executed = _asking_tool()
    parent = Agent(
        "parent",
        config=_lenient(
            "run_subtasks",
            f'{{"tasks": ["a", "b"], "parallel": {str(parallel).lower()}}}',
        ),
        tools=[ask_human],
        tasks=TaskConfig(config=_lenient("ask_human")),
    )
    stream = MemoryStream()

    with pytest.raises(HumanInputNotProvidedError):
        await parent.ask("Hi!", stream=stream)

    executed.assert_not_called()
    assert any(isinstance(event, TaskFailed) for event in await stream.history.get_events())


@pytest.mark.asyncio()
async def test_the_unanswered_call_is_closed_and_the_next_turn_works() -> None:
    """Ending the turn must not poison the conversation it ended.

    The failure escapes between a tool call and its result, so history is left
    holding an assistant tool-call nothing answers — a shape providers reject.
    Anything that reuses a stream (a ``stream=`` passed twice, an ACP or MCP
    session) would then fail on its *next* turn for a reason unrelated to the
    one it was told about.
    """
    asked = MagicMock()

    async def ask_human(ctx: Context) -> str:
        asked()
        return await ctx.input("Say smth", timeout=1.0)

    agent = Agent("", config=_lenient("ask_human"), tools=[ask_human])
    stream = MemoryStream()

    with pytest.raises(HumanInputNotProvidedError):
        await agent.ask("Hi!", stream=stream)

    events = list(await stream.history.get_events())
    called = {call.id for event in events if isinstance(event, ToolCallsEvent) for call in event.calls}
    answered = {result.parent_id for event in events if isinstance(event, ToolResultsEvent) for result in event.results}
    assert called and called == answered

    # The stand-in says what happened rather than pretending the call never
    # ran — the model reads it on the next turn.
    stand_in = next(
        result
        for event in events
        if isinstance(event, ToolResultsEvent)
        for result in event.results
        if result.parent_id in called
    )
    assert stand_in.result.parts[0].content == HUMAN_INPUT_ABANDONED_TOOL_RESULT

    # A second turn on the same history is served an answered transcript, so
    # it runs to completion instead of dying on the wreckage of the first.
    agent.config = TestConfig("recovered", raise_tool_errors=False)
    assert (await agent.ask("Still there?", stream=stream)).body == "recovered"
    asked.assert_called_once()


@pytest.mark.asyncio()
class TestSiblingToolsStopWithTheTurn:
    """The other calls in the batch do not outlive the turn they were part of.

    ``asyncio.gather`` propagates the first exception but leaves its siblings
    running detached. Nothing was watching them any more, so a side effect would
    land after the caller had already been told the turn failed — and its result
    would be written into a history that had already been repaired.

    A sync tool runs in a thread and cannot be cancelled, so the guarantee has a
    seam; the second test says exactly where it is and what survives it.
    """

    async def test_a_sibling_does_not_run_on_after_the_caller_is_told(self) -> None:
        side_effects: list[str] = []

        async def ask_human(ctx: Context) -> str:
            return await ctx.input("Say smth", timeout=1.0)

        async def slow_side_effect() -> str:
            await asyncio.sleep(0.2)
            side_effects.append("ran")
            return "done"

        agent = Agent(
            "",
            config=TestConfig(
                [ToolCallEvent(name="ask_human"), ToolCallEvent(name="slow_side_effect")],
                "done",
                raise_tool_errors=False,
            ),
            tools=[ask_human, slow_side_effect],
        )

        with pytest.raises(HumanInputNotProvidedError):
            await agent.ask("Hi!")

        # Long enough that the sibling would have finished had it been left to
        # run; the assertion is that it was stopped, not that it was slow.
        await asyncio.sleep(0.3)
        assert side_effects == []

    async def test_a_sync_sibling_is_still_never_owed_a_result(self) -> None:
        """The one case the cancellation cannot reach, pinned to what it does promise.

        A sync tool runs in a worker thread, and a thread cannot be cancelled:
        the side effect below lands after the caller has been told. What the turn
        does guarantee is that nothing the thread produces is ever used — its
        result reaches neither the stream nor the repaired history, so the model
        never reads an answer to a turn that had already failed.
        """
        finished: list[str] = []

        async def ask_human(ctx: Context) -> str:
            return await ctx.input("Say smth", timeout=1.0)

        def slow_side_effect() -> str:
            """A plain sync tool, so it runs off the event loop."""
            time.sleep(0.2)
            finished.append("ran")
            return "done"

        agent = Agent(
            "",
            config=TestConfig(
                [ToolCallEvent(name="ask_human"), ToolCallEvent(name="slow_side_effect")],
                "done",
                raise_tool_errors=False,
            ),
            tools=[ask_human, slow_side_effect],
        )
        stream = MemoryStream()

        with pytest.raises(HumanInputNotProvidedError):
            await agent.ask("Hi!", stream=stream)

        await asyncio.sleep(0.4)
        assert finished == ["ran"], "a thread cannot be cancelled; this documents that, it does not bless it"

        # ... and yet the transcript carries the stand-in, not what the thread
        # went on to return.
        events = list(await stream.history.get_events())
        stand_ins = [
            result.result.parts[0].content
            for event in events
            if isinstance(event, ToolResultsEvent)
            for result in event.results
        ]
        assert stand_ins == [HUMAN_INPUT_ABANDONED_TOOL_RESULT] * 2
        assert "done" not in stand_ins


@pytest.mark.parametrize(
    ("error", "attribute", "expected"),
    [
        (HumanInputFailedError(ApprovalQueueDownError("queue down")), "cause", ApprovalQueueDownError),
        (HumanInputTimeoutError(1.5), "timeout", 1.5),
    ],
    ids=["failed", "timeout"],
)
@pytest.mark.parametrize(
    "roundtrip",
    [copy.copy, copy.deepcopy, lambda e: pickle.loads(pickle.dumps(e))],
    ids=["copy", "deepcopy", "pickle"],
)
def test_a_channel_failure_survives_a_round_trip(
    error: HumanInputError,
    attribute: str,
    expected: object,
    roundtrip: "Callable[[HumanInputError], HumanInputError]",
) -> None:
    """These reach a caller on a ``TaskFailed`` event, so they get copied.

    The formatted message is what lands in ``args``, so the default
    reconstruction feeds it back in where the cause or the deadline goes — the
    sentence restated around itself, and the attribute that made the wrapper
    worth having replaced by a string.
    """
    restored = roundtrip(error)

    assert str(restored) == str(error)
    value = getattr(restored, attribute)
    assert isinstance(value, expected) if isinstance(expected, type) else value == expected
