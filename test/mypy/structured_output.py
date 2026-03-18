# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from typing_extensions import assert_type

from autogen.beta import Agent, ResponseSchema, response_schema
from autogen.beta.testing import TestConfig


async def check_default_response_schema() -> None:
    agent = Agent(
        "test",
        config=TestConfig(),
    )

    reply = await agent.ask("Hi, agent!")

    assert_type(reply.content, str | None)
    assert_type(await reply.validate(), str | None)


async def check_int_response_schema() -> None:
    agent = Agent(
        "test",
        config=TestConfig(),
        response_schema=int,
    )

    reply = await agent.ask("Hi, agent!")

    assert_type(reply.content, str | None)
    assert_type(await reply.validate(), int | None)


async def check_dataclass_response_schema() -> None:
    @dataclass
    class Response:
        a: int
        b: str

    agent = Agent(
        "test",
        config=TestConfig(),
        response_schema=Response,
    )

    reply = await agent.ask("Hi, agent!")
    assert_type(reply.content, str | None)
    assert_type(await reply.validate(), Response | None)


async def check_union_response_schema() -> None:
    agent = Agent(
        "test",
        config=TestConfig(),
        response_schema=int | str,
    )

    reply = await agent.ask("Hi, agent!")

    assert_type(reply.content, str | None)
    assert_type(await reply.validate(), int | str | None)


async def check_response_schema_object() -> None:
    agent = Agent(
        "test",
        config=TestConfig(),
        response_schema=ResponseSchema(int, name="Response"),
    )

    reply = await agent.ask("Hi, agent!")
    assert_type(reply.content, str | None)
    assert_type(await reply.validate(), int | None)


async def check_sync_callable_reponse() -> None:
    @response_schema
    def func(content: str) -> int:
        return int(content)

    agent = Agent(
        "test",
        config=TestConfig(),
        response_schema=func,
    )

    reply = await agent.ask("Hi, agent!")
    assert_type(reply.content, str | None)
    assert_type(await reply.validate(), int | None)


async def check_async_callable_reponse() -> None:
    @response_schema
    async def func(content: str) -> int:
        return int(content)

    agent = Agent(
        "test",
        config=TestConfig(),
        response_schema=func,
    )

    reply = await agent.ask("Hi, agent!")
    assert_type(reply.content, str | None)
    assert_type(await reply.validate(), int | None)


async def check_converstation_save_type() -> None:
    agent = Agent(
        "test",
        config=TestConfig(),
        response_schema=int,
    )

    reply = await agent.ask("Hi, agent!")
    assert_type(reply.content, str | None)
    assert_type(await reply.validate(), int | None)

    next_turn = await reply.ask("Hi, agent!")
    assert_type(next_turn.content, str | None)
    assert_type(await next_turn.validate(), int | None)


async def check_ask_overrides_response_type() -> None:
    agent = Agent("test", config=TestConfig())

    reply = await agent.ask("Hi, agent!")
    assert_type(reply.content, str | None)
    assert_type(await reply.validate(), str | None)

    reply = await agent.ask("Hi, agent!", response_schema=int)
    assert_type(reply.content, str | None)
    assert_type(await reply.validate(), int | None)


async def check_ask_none_drops_response_type() -> None:
    agent = Agent(
        "test",
        config=TestConfig(),
        response_schema=int,
    )

    reply = await agent.ask("Hi, agent!", response_schema=None)
    assert_type(reply.content, str | None)
    assert_type(await reply.validate(), str | None)


async def check_ask_response_type_not_affect_next_turn() -> None:
    agent = Agent("test", config=TestConfig(), response_schema=float)

    reply = await agent.ask("Hi, agent!")
    assert_type(reply.content, str | None)
    assert_type(await reply.validate(), float | None)

    next_turn = await reply.ask("Hi, agent!", response_schema=int)
    assert_type(next_turn.content, str | None)
    assert_type(await next_turn.validate(), int | None)

    third_turn = await next_turn.ask("Hi, agent!")
    assert_type(third_turn.content, str | None)
    assert_type(await third_turn.validate(), float | None)
