# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Any
from unittest.mock import MagicMock

import pytest

from autogen.beta import Agent, Context, observer
from autogen.beta.events import BaseEvent, HumanInputRequest, HumanMessage, ModelMessage, ModelResponse, ToolCallEvent
from autogen.beta.middleware import BaseMiddleware, Middleware
from autogen.beta.middleware.base import AgentTurn
from autogen.beta.plugin import Plugin
from autogen.beta.testing import TestConfig


class MockClient:
    """Minimal LLM client that records which prompt was active and returns a fixed reply."""

    def __init__(self, mock: MagicMock) -> None:
        self.mock = mock

    def copy(self) -> "MockClient":
        return self

    def create(self) -> "MockClient":
        return self

    async def __call__(
        self,
        messages: Sequence[BaseEvent],
        context: Context,
        **kwargs: Any,
    ) -> ModelResponse:
        self.mock(context.prompt)
        return ModelResponse(ModelMessage("reply"))


@pytest.mark.asyncio
class TestPluginTools:
    async def test_via_constructor(self, mock: MagicMock) -> None:
        test_config = TestConfig(ToolCallEvent(name="my_tool"), "result")

        def my_tool(ctx: Context) -> str:
            mock()
            return "ok"

        plugin = Plugin(tools=[my_tool])
        agent = Agent("agent", config=test_config, plugins=[plugin])

        await agent.ask("Hi!")
        mock.assert_called_once()

    async def test_decorator(self, mock: MagicMock) -> None:
        test_config = TestConfig(ToolCallEvent(name="tool2"), "result")

        plugin = Plugin()

        @plugin.tool
        def tool2(ctx: Context) -> str:
            mock()
            return "ok"

        agent = Agent("agent", config=test_config, plugins=[plugin])
        await agent.ask("Hi!")
        mock.assert_called_once()

    async def test_combined_with_agent_tools(self, mock: MagicMock) -> None:
        """Tools from plugins and from the agent itself are both registered."""
        test_config = TestConfig(ToolCallEvent(name="plugin_tool"), "result")

        def plugin_tool(ctx: Context) -> str:
            mock.plugin()
            return "ok"

        def agent_tool(ctx: Context) -> str:  # noqa: ARG001
            mock.agent()
            return "ok"

        plugin = Plugin(tools=[plugin_tool])
        agent = Agent("agent", config=test_config, plugins=[plugin], tools=[agent_tool])

        # Only plugin_tool is triggered by TestConfig here, but both should be registered
        await agent.ask("Hi!")
        mock.plugin.assert_called_once()
        mock.agent.assert_not_called()


@pytest.mark.asyncio
class TestPluginPrompts:
    async def test_static(self, mock: MagicMock) -> None:
        plugin = Plugin(prompt="from plugin")
        agent = Agent("agent", config=MockClient(mock), plugins=[plugin])

        await agent.ask("Hi!")
        mock.assert_called_once_with(["from plugin"])

    async def test_dynamic(self, mock: MagicMock) -> None:
        plugin = Plugin()

        @plugin.prompt
        async def my_prompt() -> str:
            return "dynamic"

        agent = Agent("agent", config=MockClient(mock), plugins=[plugin])
        await agent.ask("Hi!")
        mock.assert_called_once_with(["dynamic"])

    async def test_multiple_plugins_ordered(self, mock: MagicMock) -> None:
        p1 = Plugin(prompt="first")
        p2 = Plugin(prompt="second")
        agent = Agent("agent", config=MockClient(mock), plugins=[p1, p2])

        await agent.ask("Hi!")
        mock.assert_called_once_with(["first", "second"])


@pytest.mark.asyncio
class TestPluginObservers:
    async def test_via_constructor(self, mock: MagicMock) -> None:
        test_config = TestConfig("response")

        plugin = Plugin(observers=[observer(ModelResponse, mock)])
        agent = Agent("agent", config=test_config, plugins=[plugin])

        await agent.ask("Hi!")
        mock.assert_called_once()

    async def test_decorator(self, mock: MagicMock) -> None:
        test_config = TestConfig("response")

        plugin = Plugin()

        @plugin.observer(ModelResponse)
        def on_response(event: ModelResponse) -> None:
            mock()

        agent = Agent("agent", config=test_config, plugins=[plugin])
        await agent.ask("Hi!")
        mock.assert_called_once()


@pytest.mark.asyncio
class TestPluginDependenciesAndVariables:
    async def test_dependencies_available_in_tool(self, mock: MagicMock) -> None:
        test_config = TestConfig(ToolCallEvent(name="my_tool"), "result")

        dep_value = object()

        def my_tool(ctx: Context) -> str:
            mock(ctx.dependencies["dep"])
            return "ok"

        plugin = Plugin(tools=[my_tool], dependencies={"dep": dep_value})
        agent = Agent("agent", config=test_config, plugins=[plugin])

        await agent.ask("Hi!")
        mock.assert_called_once_with(dep_value)

    async def test_variables_available_in_tool(self, mock: MagicMock) -> None:
        test_config = TestConfig(ToolCallEvent(name="my_tool"), "result")

        def my_tool(ctx: Context) -> str:
            mock(ctx.variables["var"])
            return "ok"

        plugin = Plugin(tools=[my_tool], variables={"var": "hello"})
        agent = Agent("agent", config=test_config, plugins=[plugin])

        await agent.ask("Hi!")
        mock.assert_called_once_with("hello")

    async def test_agent_dependencies_override_plugin(self, mock: MagicMock) -> None:
        test_config = TestConfig(ToolCallEvent(name="my_tool"), "result")

        def my_tool(ctx: Context) -> str:
            mock(ctx.dependencies["dep"])
            return "ok"

        plugin = Plugin(tools=[my_tool], dependencies={"dep": "from_plugin"})
        agent = Agent("agent", config=test_config, plugins=[plugin], dependencies={"dep": "from_agent"})

        await agent.ask("Hi!")
        mock.assert_called_once_with("from_agent")


@pytest.mark.asyncio
class TestPluginHITLHook:
    async def test_via_constructor(self, mock: MagicMock) -> None:
        test_config = TestConfig(ToolCallEvent(name="my_tool"), "result")

        async def my_tool(ctx: Context) -> str:
            mock(await ctx.input("prompt", timeout=1.0))
            return "ok"

        def hitl(event: HumanInputRequest) -> HumanMessage:
            return HumanMessage("answer")

        plugin = Plugin(tools=[my_tool], hitl_hook=hitl)
        agent = Agent("agent", config=test_config, plugins=[plugin])

        await agent.ask("Hi!")
        mock.assert_called_once_with("answer")

    async def test_decorator(self, mock: MagicMock) -> None:
        test_config = TestConfig(ToolCallEvent(name="my_tool"), "result")

        async def my_tool(ctx: Context) -> str:
            mock(await ctx.input("prompt", timeout=1.0))
            return "ok"

        plugin = Plugin(tools=[my_tool])

        @plugin.hitl_hook
        def hitl(event: HumanInputRequest) -> HumanMessage:
            return HumanMessage("from_decorator")

        agent = Agent("agent", config=test_config, plugins=[plugin])
        await agent.ask("Hi!")
        mock.assert_called_once_with("from_decorator")

    async def test_agent_hook_overrides_plugin(self, mock: MagicMock) -> None:
        """Agent's direct hitl_hook takes priority over the plugin's."""
        test_config = TestConfig(ToolCallEvent(name="my_tool"), "result")

        async def my_tool(ctx: Context) -> str:
            mock(await ctx.input("prompt", timeout=1.0))
            return "ok"

        def plugin_hitl(event: HumanInputRequest) -> HumanMessage:
            return HumanMessage("plugin_answer")

        def agent_hitl(event: HumanInputRequest) -> HumanMessage:
            return HumanMessage("agent_answer")

        plugin = Plugin(tools=[my_tool], hitl_hook=plugin_hitl)
        agent = Agent("agent", config=test_config, plugins=[plugin], hitl_hook=agent_hitl)

        await agent.ask("Hi!")
        mock.assert_called_once_with("agent_answer")

    def test_warn_on_double_set(self) -> None:
        plugin = Plugin()

        @plugin.hitl_hook
        def hook1(event: HumanInputRequest) -> HumanMessage:
            return HumanMessage("1")

        with pytest.warns(RuntimeWarning):

            @plugin.hitl_hook
            def hook2(event: HumanInputRequest) -> HumanMessage:
                return HumanMessage("2")


@pytest.mark.asyncio
async def test_plugin_middleware_is_invoked(mock: MagicMock) -> None:
    class TrackingMiddleware(BaseMiddleware):
        async def on_turn(self, call_next: AgentTurn, event: BaseEvent, context: Context) -> ModelResponse:
            mock()
            return await call_next(event, context)

    test_config = TestConfig("response")
    plugin = Plugin(middleware=[Middleware(TrackingMiddleware)])
    agent = Agent("agent", config=test_config, plugins=[plugin])

    await agent.ask("Hi!")
    mock.assert_called_once()


@pytest.mark.asyncio
class TestMultiplePlugins:
    async def test_tools_all_registered(self, mock: MagicMock) -> None:
        test_config = TestConfig(ToolCallEvent(name="tool_a"), "result")

        def tool_a(ctx: Context) -> str:
            mock.a()
            return "ok"

        def tool_b(ctx: Context) -> str:  # noqa: ARG001
            mock.b()
            return "ok"

        p1 = Plugin(tools=[tool_a])
        p2 = Plugin(tools=[tool_b])
        agent = Agent("agent", config=test_config, plugins=[p1, p2])

        await agent.ask("Hi!")
        mock.a.assert_called_once()
        mock.b.assert_not_called()

    async def test_first_plugin_hitl_wins_and_warns_on_conflict(self, mock: MagicMock) -> None:
        """When multiple plugins set hitl_hook and the agent doesn't, the first plugin wins and a warning is emitted."""
        test_config = TestConfig(ToolCallEvent(name="my_tool"), "result")

        async def my_tool(ctx: Context) -> str:
            mock(await ctx.input("prompt", timeout=1.0))
            return "ok"

        def hook1(event: HumanInputRequest) -> HumanMessage:
            return HumanMessage("first")

        def hook2(event: HumanInputRequest) -> HumanMessage:
            return HumanMessage("second")

        p1 = Plugin(tools=[my_tool], hitl_hook=hook1)
        p2 = Plugin(hitl_hook=hook2)

        with pytest.warns(UserWarning, match="already has a HITL hook"):
            agent = Agent("agent", config=test_config, plugins=[p1, p2])

        await agent.ask("Hi!")
        mock.assert_called_once_with("first")
