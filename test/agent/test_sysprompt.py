# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from ag2 import Agent, Context, MemoryStream
from ag2.events import BaseEvent
from ag2.testing import TestConfig
from test._helpers import LLMCalls


class CustomEvent(BaseEvent):
    pass


@pytest.mark.asyncio()
async def test_sysprompt() -> None:
    calls = LLMCalls()
    agent = Agent(
        "test",
        prompt="You are a helpful agent!",
        config=TestConfig("Hi, user!"),
        middleware=[calls.middleware()],
    )

    conversation = await agent.ask("Hi, agent!")

    assert calls.prompts == [["You are a helpful agent!"]]
    assert conversation.context.prompt == ["You are a helpful agent!"]


@pytest.mark.asyncio()
async def test_multiple_sysprompts() -> None:
    calls = LLMCalls()
    agent = Agent(
        "test",
        prompt=["1", "2"],
        config=TestConfig("Hi, user!"),
        middleware=[calls.middleware()],
    )

    conversation = await agent.ask("Hi, agent!")

    assert calls.prompts == [["1", "2"]]
    assert conversation.context.prompt == ["1", "2"]


@pytest.mark.asyncio()
async def test_sysprompt_reuse() -> None:
    calls = LLMCalls()
    agent = Agent(
        "test",
        prompt="You are a helpful agent!",
        config=TestConfig("Hi, user!", "Hi, user!"),
        middleware=[calls.middleware()],
    )

    conversation = await agent.ask("Hi, agent!")
    await conversation.ask("Next turn")

    assert calls.prompts == [["You are a helpful agent!"], ["You are a helpful agent!"]]


@pytest.mark.asyncio()
async def test_sysprompt_override_with_call() -> None:
    calls = LLMCalls()
    agent = Agent(
        "test",
        prompt="You are a helpful agent!",
        config=TestConfig("Hi, user!"),
        middleware=[calls.middleware()],
    )

    await agent.ask("Hi, agent!", prompt=["1"])
    assert calls.prompts == [["1"]]


@pytest.mark.asyncio()
async def test_callable_sysprompt() -> None:
    calls = LLMCalls()

    async def sysprompt() -> str:
        return "1"

    agent = Agent(
        "test",
        prompt=sysprompt,
        config=TestConfig("Hi, user!"),
        middleware=[calls.middleware()],
    )

    await agent.ask("Hi, agent!")
    assert calls.prompts == [["1"]]


@pytest.mark.asyncio()
async def test_callable_sysprompt_called_once(mock: MagicMock) -> None:
    async def sysprompt(event: BaseEvent, ctx: Context) -> str:
        mock.prompt()
        return "1"

    agent = Agent(
        "test",
        prompt=sysprompt,
        config=TestConfig("Hi, user!", "Hi, user!"),
    )

    conversation = await agent.ask("Hi, agent!")
    await conversation.ask("Next turn")

    mock.prompt.assert_called_once()


@pytest.mark.asyncio()
async def test_decorator_sysprompt() -> None:
    calls = LLMCalls()
    agent = Agent("test", config=TestConfig("Hi, user!"), middleware=[calls.middleware()])

    @agent.prompt
    async def sysprompt(event: BaseEvent, ctx: Context) -> str:
        return "1"

    await agent.ask("Hi, agent!")
    assert calls.prompts == [["1"]]


@pytest.mark.asyncio()
async def test_callable_sysprompt_decorator() -> None:
    calls = LLMCalls()
    agent = Agent("test", config=TestConfig("Hi, user!"), middleware=[calls.middleware()])

    @agent.prompt()
    def sysprompt(ctx: Context) -> str:
        return "1"

    await agent.ask("Hi, agent!")
    assert calls.prompts == [["1"]]


@pytest.mark.asyncio()
async def test_mixed_sysprompts() -> None:
    calls = LLMCalls()

    async def sysprompt(event: BaseEvent, ctx: Context) -> str:
        assert ctx.prompt == ["1"]
        return "2"

    agent = Agent(
        "test",
        prompt=["1", sysprompt],
        config=TestConfig("Hi, user!"),
        middleware=[calls.middleware()],
    )

    await agent.ask("Hi, agent!")

    assert calls.prompts == [["1", "2"]]


@pytest.mark.asyncio()
async def test_prompt_mutation() -> None:
    calls = LLMCalls()
    agent = Agent(
        "test",
        prompt="1",
        config=TestConfig("Hi, user!", "Hi, user!"),
        middleware=[calls.middleware()],
    )

    # test first call
    conversation = await agent.ask("Hi, agent!")
    assert calls.prompts == [["1"]]

    # test second call
    conversation.context.prompt = ["2"]
    await conversation.ask("Next turn")

    assert calls.prompts == [["1"], ["2"]]


@pytest.mark.asyncio()
async def test_prompt_mutation_from_subscriber() -> None:
    agent = Agent(
        "test",
        prompt="1",
        # Published mid-call, where a provider streams chunks.
        config=TestConfig(CustomEvent(), "Hi, user!"),
    )

    stream = MemoryStream()

    @stream.where(CustomEvent).subscribe()
    async def mutate_prompt(event: CustomEvent, ctx: Context) -> None:
        assert ctx.prompt == ["1"]
        ctx.prompt = ["2"]

    reply = await agent.ask("Hi, agent!", stream=stream)
    assert reply.context.prompt == ["2"]
