from typing import Any
from unittest.mock import MagicMock

import pytest

from autogen.beta import Agent, Context, MemoryStream
from autogen.beta.config import LLMClient
from autogen.beta.events import BaseEvent, ModelResponse


class CustomEvent(BaseEvent):
    pass


class MockClient(LLMClient):
    def __init__(self, mock: MagicMock) -> None:
        self.mock = mock

    def create(self) -> "MockClient":
        return self

    async def __call__(self, *messages: BaseEvent, ctx: Context, **kwargs: Any) -> None:
        await ctx.send(CustomEvent())
        self.mock(ctx.prompt)
        await ctx.send(ModelResponse(response="Hi, user!"))


@pytest.mark.asyncio()
async def test_sysprompt(mock: MagicMock):
    agent = Agent(
        "test",
        prompt="You are a helpful agent!",
        config=MockClient(mock),
    )

    conversation = await agent.ask("Hi, agent!")

    mock.assert_called_once_with(["You are a helpful agent!"])
    assert conversation.ctx.prompt == ["You are a helpful agent!"]


@pytest.mark.asyncio()
async def test_multiple_sysprompts(mock: MagicMock):
    agent = Agent(
        "test",
        prompt=["1", "2"],
        config=MockClient(mock),
    )

    conversation = await agent.ask("Hi, agent!")

    mock.assert_called_once_with(["1", "2"])
    assert conversation.ctx.prompt == ["1", "2"]


@pytest.mark.asyncio()
async def test_sysprompt_reuse(mock: MagicMock):
    agent = Agent(
        "test",
        prompt="You are a helpful agent!",
        config=MockClient(mock),
    )

    conversation = await agent.ask("Hi, agent!")
    await conversation.ask("Next turn")

    mock.assert_called_with(["You are a helpful agent!"])
    assert mock.call_count == 2


@pytest.mark.asyncio()
async def test_sysprompt_override_with_call(mock: MagicMock):
    agent = Agent(
        "test",
        prompt="You are a helpful agent!",
        config=MockClient(mock),
    )

    await agent.ask("Hi, agent!", prompt=["1"])
    mock.assert_called_once_with(["1"])


@pytest.mark.asyncio()
async def test_callable_sysprompt(mock: MagicMock):
    async def sysprompt(event: BaseEvent, ctx: Context) -> str:
        return "1"

    agent = Agent(
        "test",
        prompt=sysprompt,
        config=MockClient(mock),
    )

    await agent.ask("Hi, agent!")
    mock.assert_called_once_with(["1"])


@pytest.mark.asyncio()
async def test_callable_sysprompt_called_once(mock: MagicMock):
    async def sysprompt(event: BaseEvent, ctx: Context) -> str:
        mock.prompt()
        return "1"

    agent = Agent(
        "test",
        prompt=sysprompt,
        config=MockClient(mock),
    )

    conversation = await agent.ask("Hi, agent!")
    await conversation.ask("Next turn")

    mock.prompt.assert_called_once()


@pytest.mark.asyncio()
async def test_decorator_sysprompt(mock: MagicMock):
    agent = Agent(
        "test",
        config=MockClient(mock),
    )

    @agent.prompt
    async def sysprompt(event: BaseEvent, ctx: Context) -> str:
        return "1"

    await agent.ask("Hi, agent!")
    mock.assert_called_once_with(["1"])


@pytest.mark.asyncio()
async def test_mixed_sysprompts(mock: MagicMock):
    async def sysprompt(event: BaseEvent, ctx: Context) -> str:
        assert ctx.prompt == ["1"]
        return "2"

    agent = Agent(
        "test",
        prompt=["1", sysprompt],
        config=MockClient(mock),
    )

    await agent.ask("Hi, agent!")

    mock.assert_called_once_with(["1", "2"])


@pytest.mark.asyncio()
async def test_prompt_mutation(mock: MagicMock):
    agent = Agent(
        "test",
        prompt="1",
        config=MockClient(mock),
    )

    # test first call
    conversation = await agent.ask("Hi, agent!")
    mock.assert_called_once_with(["1"])

    # test second call
    conversation.ctx.prompt = ["2"]
    await conversation.ask("Next turn")

    # validate latest call
    mock.assert_called_with(["2"])

    # validate all calls
    assert [c[0][0] for c in mock.call_args_list] == [
        ["1"],
        ["2"],
    ]


@pytest.mark.asyncio()
async def test_prompt_mutation_from_subscriber(mock: MagicMock):
    agent = Agent(
        "test",
        prompt="1",
        config=MockClient(mock),
    )

    stream = MemoryStream()

    @stream.where(CustomEvent).subscribe()
    async def mutate_prompt(event: CustomEvent, ctx: Context) -> None:
        assert ctx.prompt == ["1"]
        ctx.prompt = ["2"]

    await agent.ask("Hi, agent!", stream=stream)
    mock.assert_called_once_with(["2"])
