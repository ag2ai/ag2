# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from unittest.mock import MagicMock

import pytest

from ag2 import Agent, Context
from ag2.events import BaseEvent, ModelResponse, ToolCallEvent
from ag2.middleware import BaseMiddleware, LLMCall, Middleware, ToolChoice, ToolChoiceMode
from ag2.testing import ModelCall, TestConfig, TrackingConfig
from ag2.tools.final import FunctionToolSchema


class ChoosingMiddleware(BaseMiddleware):
    def __init__(
        self,
        event: BaseEvent,
        ctx: Context,
        choice: ToolChoice | ToolChoiceMode,
        first_call_only: bool = False,
    ) -> None:
        super().__init__(event, ctx)
        self.choice = choice
        self.first_call_only = first_call_only
        self.calls = 0

    async def on_llm_call(self, call_next: LLMCall, events: Sequence[BaseEvent], ctx: Context) -> ModelResponse:
        self.calls += 1
        if self.first_call_only and self.calls > 1:
            return await call_next(events, ctx)
        return await call_next(events, ctx.with_tool_choice(self.choice))


class SeeingMiddleware(BaseMiddleware):
    def __init__(self, event: BaseEvent, ctx: Context, mock: MagicMock) -> None:
        super().__init__(event, ctx)
        self.mock = mock

    async def on_llm_call(self, call_next: LLMCall, events: Sequence[BaseEvent], ctx: Context) -> ModelResponse:
        self.mock(ctx.tool_choice)
        return await call_next(events, ctx)


def _agent(config: TrackingConfig, *middleware: Middleware) -> Agent:
    agent = Agent("agent", config=config, middleware=middleware)

    @agent.tool
    def search() -> str:
        return "found"

    @agent.tool
    def finish() -> str:
        return "finished"

    @agent.tool
    def route() -> str:
        return "routed"

    return agent


def _tool_names(call: ModelCall) -> list[str]:
    return [t.function.name for t in call.tools if isinstance(t, FunctionToolSchema)]


@pytest.mark.asyncio()
class TestToolChoiceOnLLMCall:
    async def test_none_keeps_tools_and_reaches_the_client(self) -> None:
        config = TrackingConfig(TestConfig("answer"))
        agent = _agent(config, Middleware(ChoosingMiddleware, choice="none"))

        await agent.ask("Hi!")

        (call,) = config.calls
        assert call.tool_choice == ToolChoice("none")
        assert _tool_names(call) == ["search", "finish", "route"]

    async def test_allow_list_filters_tools(self) -> None:
        config = TrackingConfig(TestConfig("answer"))
        agent = _agent(config, Middleware(ChoosingMiddleware, choice=ToolChoice.only("finish", "route")))

        await agent.ask("Hi!")

        (call,) = config.calls
        assert _tool_names(call) == ["finish", "route"]
        assert call.tool_choice == ToolChoice.only("finish", "route")

    async def test_allow_list_of_absent_tools_is_none(self) -> None:
        config = TrackingConfig(TestConfig("answer"))
        agent = _agent(config, Middleware(ChoosingMiddleware, choice=ToolChoice.only("missing")))

        await agent.ask("Hi!")

        (call,) = config.calls
        assert call.tool_choice == ToolChoice("none")
        assert _tool_names(call) == ["search", "finish", "route"]

    async def test_applies_to_one_call_only(self) -> None:
        config = TrackingConfig(TestConfig(ToolCallEvent(name="search"), "answer"))
        agent = _agent(config, Middleware(ChoosingMiddleware, choice="required", first_call_only=True))

        await agent.ask("Hi!")

        assert [c.tool_choice for c in config.calls] == [ToolChoice("required"), None]

    async def test_no_choice_leaves_the_call_unchanged(self) -> None:
        config = TrackingConfig(TestConfig("answer"))
        agent = _agent(config)

        await agent.ask("Hi!")

        (call,) = config.calls
        assert call.tool_choice is None
        assert _tool_names(call) == ["search", "finish", "route"]

    async def test_inner_middleware_runs_and_sees_choice(self, mock: MagicMock) -> None:
        config = TrackingConfig(TestConfig("answer"))
        agent = _agent(
            config,
            Middleware(ChoosingMiddleware, choice="none"),
            Middleware(SeeingMiddleware, mock=mock),
        )

        await agent.ask("Hi!")

        mock.assert_called_once_with(ToolChoice("none"))

    async def test_nested_allow_lists_intersect(self) -> None:
        config = TrackingConfig(TestConfig("answer"))
        agent = _agent(
            config,
            Middleware(ChoosingMiddleware, choice=ToolChoice.only("search", "finish")),
            Middleware(ChoosingMiddleware, choice=ToolChoice.only("finish", "route")),
        )

        await agent.ask("Hi!")

        (call,) = config.calls
        assert _tool_names(call) == ["finish"]

    async def test_inner_choice_cannot_widen_outer_none(self) -> None:
        config = TrackingConfig(TestConfig("answer"))
        agent = _agent(
            config,
            Middleware(ChoosingMiddleware, choice="none"),
            Middleware(ChoosingMiddleware, choice=ToolChoice.only("finish")),
        )

        await agent.ask("Hi!")

        (call,) = config.calls
        assert call.tool_choice == ToolChoice("none")

    @pytest.mark.parametrize("choice", ["none", ToolChoice.only("finish")])
    async def test_call_outside_choice_is_not_executed(
        self, choice: ToolChoice | ToolChoiceMode, mock: MagicMock
    ) -> None:
        # The model names a tool the call did not offer, as a model or a gateway can.
        agent = Agent(
            "agent",
            config=TestConfig(ToolCallEvent(name="search"), "answer"),
            middleware=[Middleware(ChoosingMiddleware, choice=choice)],
        )

        @agent.tool
        def search() -> str:
            mock()
            return "found"

        @agent.tool
        def finish() -> str:
            return "finished"

        reply = await agent.ask("Hi!")

        mock.assert_not_called()
        assert reply.body is None

    async def test_allowed_call_is_executed(self, mock: MagicMock) -> None:
        agent = Agent(
            "agent",
            config=TestConfig(ToolCallEvent(name="finish"), "answer"),
            middleware=[Middleware(ChoosingMiddleware, choice=ToolChoice.only("finish"), first_call_only=True)],
        )

        @agent.tool
        def finish() -> str:
            mock()
            return "finished"

        reply = await agent.ask("Hi!")

        mock.assert_called_once()
        assert reply.body == "answer"

    async def test_required_without_tools_raises(self) -> None:
        agent = Agent(
            "agent",
            config=TestConfig("answer"),
            middleware=[Middleware(ChoosingMiddleware, choice="required")],
        )

        with pytest.raises(ValueError, match="required"):
            await agent.ask("Hi!")


class TestToolChoice:
    def test_none_drops_allow_list(self) -> None:
        assert ToolChoice("none", frozenset({"a"})) == ToolChoice("none")

    def test_empty_allow_list_is_none(self) -> None:
        assert ToolChoice.only() == ToolChoice("none")

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown tool choice mode"):
            ToolChoice("never")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("outer", "inner", "expected"),
        [
            ("auto", "required", ToolChoice("required")),
            ("required", "auto", ToolChoice("required")),
            ("none", "required", ToolChoice("none")),
            ("required", "none", ToolChoice("none")),
            (ToolChoice.only("a", "b"), ToolChoice.only("b", "c"), ToolChoice.only("b")),
            (ToolChoice.only("a"), ToolChoice.only("b"), ToolChoice("none")),
            (ToolChoice.only("a"), "required", ToolChoice.only("a", mode="required")),
            ("auto", ToolChoice.only("a"), ToolChoice.only("a")),
        ],
    )
    def test_narrow_never_widens(
        self,
        outer: ToolChoice | ToolChoiceMode,
        inner: ToolChoice | ToolChoiceMode,
        expected: ToolChoice,
    ) -> None:
        assert ToolChoice.ensure(outer).narrow(inner) == expected
