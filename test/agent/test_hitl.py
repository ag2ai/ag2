# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import MagicMock

import pytest

from ag2 import Agent, Context
from ag2.events import HumanInputRequest, HumanMessage, ToolCallEvent
from ag2.exceptions import HumanInputFailedError, HumanInputNotProvidedError, HumanInputTimeoutError
from ag2.middleware import approval_required
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


@pytest.mark.asyncio()
class TestChannelFailureCrossesTheSubtaskBoundary:
    """A sub-agent's unanswerable question is the same failure one level down.

    Delegation turns a child's exception into the delegating tool's output, so
    without this the parent's model reads "the subtask failed" and is invited to
    find another way to the same effect.
    """

    def _parent_with_asking_child(self) -> tuple[Agent, MagicMock]:
        executed = MagicMock()

        async def ask_human(ctx: Context) -> str:
            answer = await ctx.input("Say smth", timeout=1.0)
            executed()
            return answer

        child = Agent("child", config=_lenient("ask_human"), tools=[ask_human])
        parent = Agent(
            "parent",
            config=_lenient("delegate", '{"objective": "ask the human"}'),
            tools=[subagent_tool(child, name="delegate", description="delegate to the child")],
        )
        return parent, executed

    async def test_a_delegated_question_nobody_can_answer_ends_the_parent_turn(self) -> None:
        parent, executed = self._parent_with_asking_child()

        with pytest.raises(HumanInputNotProvidedError):
            await parent.ask("Hi!")

        executed.assert_not_called()
