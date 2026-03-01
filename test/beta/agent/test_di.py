from typing import Annotated, Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from autogen.beta import Agent, Context, Inject, MemoryStream
from autogen.beta.config import LLMClient
from autogen.beta.events import BaseEvent, ModelMessage, ModelResponse, ToolCall, ToolCalls, ToolError


class TestConfig(LLMClient):
    __test__ = False

    def __init__(self, *events: BaseEvent) -> None:
        self.events = iter(events)

    def create(self) -> "TestConfig":
        return self

    async def __call__(self, *messages: BaseEvent, ctx: Context, **kwargs: Any) -> None:
        await ctx.send(next(self.events))


@pytest.fixture()
def test_config() -> TestConfig:
    return TestConfig(
        ModelResponse(
            message=None,
            tool_calls=ToolCalls(
                calls=[
                    ToolCall(
                        id=uuid4(),
                        name="my_tool",
                        arguments="{}",
                    )
                ]
            ),
            usage={},
        ),
        ModelResponse(
            message=ModelMessage(content="result"),
            tool_calls=[],
            usage={},
        ),
    )


@pytest.mark.asyncio()
async def test_call_tool_with_injected_object(
    mock: MagicMock,
    test_config: TestConfig,
) -> None:
    def my_tool(ctx: Context) -> str:
        mock(ctx.container["dep"])
        return ""

    agent = Agent(
        "",
        config=test_config,
        tools=[my_tool],
    )

    dependency = object()

    await agent.ask("Hi!", dependencies={"dep": dependency})

    mock.assert_called_once_with(dependency)


@pytest.mark.asyncio()
async def test_call_tool_with_agent_dependency(
    mock: MagicMock,
    test_config: TestConfig,
) -> None:
    def my_tool(ctx: Context) -> str:
        mock(ctx.container["dep"])
        return ""

    dependency = object()

    agent = Agent(
        "",
        config=test_config,
        tools=[my_tool],
        dependencies={"dep": dependency},
    )

    await agent.ask("Hi!")

    mock.assert_called_once_with(dependency)


@pytest.mark.asyncio()
async def test_inject_alias(
    mock: MagicMock,
    test_config: TestConfig,
) -> None:
    def my_tool(
        dep: Annotated[str, Inject()],
    ) -> str:
        mock(dep)
        return ""

    dependency = object()

    agent = Agent(
        "",
        config=test_config,
        tools=[my_tool],
        dependencies={"dep": dependency},
    )

    await agent.ask("Hi!")
    mock.assert_called_once_with(dependency)


@pytest.mark.asyncio()
async def test_inject_by_custon_name(
    mock: MagicMock,
    test_config: TestConfig,
) -> None:
    def my_tool(
        d: Annotated[str, Inject("dep")],
    ) -> str:
        mock(d)
        return ""

    dependency = object()

    agent = Agent(
        "",
        config=test_config,
        tools=[my_tool],
        dependencies={"dep": dependency},
    )

    await agent.ask("Hi!")
    mock.assert_called_once_with(dependency)


@pytest.mark.asyncio()
async def test_inject_with_default(
    mock: MagicMock,
    test_config: TestConfig,
) -> None:
    def my_tool(
        dep: Annotated[str, Inject(default=1)],
    ) -> str:
        mock(dep)
        return ""

    agent = Agent(
        "",
        config=test_config,
        tools=[my_tool],
    )

    await agent.ask("Hi!")
    mock.assert_called_once_with(1)


@pytest.mark.asyncio()
async def test_miss_injection(
    mock: MagicMock,
    test_config: TestConfig,
) -> None:
    def my_tool(
        dep: Annotated[str, Inject()],
    ) -> str:
        return dep

    agent = Agent(
        "",
        config=test_config,
        tools=[my_tool],
    )

    stream = MemoryStream()

    @stream.where(ToolError).subscribe()
    async def catch_error(ev: ToolError, ctx: Context) -> None:
        mock("Field required" in ev.content)

    await agent.ask("Hi!", stream=stream)
    mock.assert_called_once_with(True)
