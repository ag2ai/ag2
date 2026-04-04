# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from autogen.beta import Agent, PromptedSchema, ResponseSchema, response_schema
from autogen.beta.testing import TestConfig, TrackingConfig

# --- Agent-level response_schema ---


@pytest.mark.asyncio()
class TestAgentLevelResponseSchema:
    async def test_type_response_schema(self) -> None:
        agent = Agent("test", config=TestConfig('{"data": 42}'), response_schema=int)

        reply = await agent.ask("Hi!")
        result = await reply.content()

        assert result == 42

    async def test_dataclass_response_schema(self) -> None:
        @dataclass
        class Point:
            x: float
            y: float

        agent = Agent("test", config=TestConfig('{"x": 1.5, "y": 2.5}'), response_schema=Point)

        reply = await agent.ask("Hi!")
        result = await reply.content()

        assert result == Point(x=1.5, y=2.5)

    async def test_pydantic_response_schema(self) -> None:
        class User(BaseModel):
            name: str
            age: int

        agent = Agent("test", config=TestConfig('{"name": "Alice", "age": 30}'), response_schema=User)

        reply = await agent.ask("Hi!")
        result = await reply.content()

        assert isinstance(result, User)
        assert result.name == "Alice"
        assert result.age == 30

    async def test_response_schema_object(self) -> None:
        schema = ResponseSchema(int, name="MyInt")
        agent = Agent("test", config=TestConfig('{"data": 42}'), response_schema=schema)

        reply = await agent.ask("Hi!")
        result = await reply.content()

        assert result == 42

    async def test_callable_response_schema(self) -> None:
        @response_schema
        def double(content: str) -> int:
            return int(content) * 2

        agent = Agent("test", config=TestConfig("21"), response_schema=double)

        reply = await agent.ask("Hi!")
        result = await reply.content()

        assert result == 42

    async def test_async_callable_response_schema(self) -> None:
        @response_schema
        async def double(content: str) -> int:
            return int(content) * 2

        agent = Agent("test", config=TestConfig("21"), response_schema=double)

        reply = await agent.ask("Hi!")
        result = await reply.content()

        assert result == 42

    async def test_prompted_schema_with_type(self) -> None:
        agent = Agent("test", config=TestConfig('{"data": 42}'), response_schema=PromptedSchema(int))

        reply = await agent.ask("Hi!")
        result = await reply.content()

        assert result == 42

    async def test_prompted_schema_with_response_schema(self) -> None:
        inner = ResponseSchema(int, name="MyInt")
        agent = Agent("test", config=TestConfig('{"data": 42}'), response_schema=PromptedSchema(inner))

        reply = await agent.ask("Hi!")
        result = await reply.content()

        assert result == 42

    async def test_prompted_schema_with_callable(self) -> None:
        @response_schema
        def double(content: str) -> int:
            return int(content) * 2

        agent = Agent("test", config=TestConfig("21"), response_schema=PromptedSchema(double))

        reply = await agent.ask("Hi!")
        result = await reply.content()

        assert result == 42

    async def test_no_schema_returns_string(self) -> None:
        agent = Agent("test", config=TestConfig("hello"))

        reply = await agent.ask("Hi!")
        result = await reply.content()

        assert result == "hello"

    async def test_validation_error(self) -> None:
        agent = Agent("test", config=TestConfig("not a number"), response_schema=int)

        reply = await agent.ask("Hi!")

        with pytest.raises(Exception):
            await reply.content()

    async def test_retry_succeeds_on_second_attempt(self) -> None:
        tracking = TrackingConfig(TestConfig("not a number", '{"data": 42}'))
        agent = Agent("test", config=tracking, response_schema=int)

        reply = await agent.ask("Hi!")

        result = await reply.content(retries=1)
        assert result == 42

        # 1 initial ask + 1 retry
        assert tracking.mock.call_count == 2
        retry_msg = tracking.mock.call_args_list[1][0][0]
        assert "not a number" in retry_msg.content

    async def test_retry_with_prompted_schema_omits_null_schema(self) -> None:
        tracking = TrackingConfig(TestConfig("not a number", '{"data": 42}'))
        agent = Agent("test", config=tracking, response_schema=PromptedSchema(int))

        reply = await agent.ask("Hi!")

        result = await reply.content(retries=1)
        assert result == 42

        retry_msg = tracking.mock.call_args_list[1][0][0]
        assert "null" not in retry_msg.content

    async def test_retry_exhausted_raises(self) -> None:
        tracking = TrackingConfig(TestConfig("bad", "still bad"))
        agent = Agent("test", config=tracking, response_schema=int)

        reply = await agent.ask("Hi!")

        with pytest.raises(Exception):
            await reply.content(retries=1)

        # 1 initial ask + 1 retry (both fail)
        assert tracking.mock.call_count == 2

    async def test_retries_keeps_retrying(self) -> None:
        tracking = TrackingConfig(TestConfig("bad", "bad", "bad", '{"data": 7}'))
        agent = Agent("test", config=tracking, response_schema=int)

        reply = await agent.ask("Hi!")
        result = await reply.content(retries=math.inf)

        assert result == 7
        # 1 initial ask + 3 retries
        assert tracking.mock.call_count == 4


# --- Ask-level response_schema ---
@pytest.mark.asyncio()
class TestAskLevelResponseSchema:
    async def test_ask_type_override(self) -> None:
        agent = Agent("test", config=TestConfig('{"data": 42}'))

        reply = await agent.ask("Hi!", response_schema=int)
        result = await reply.content()

        assert result == 42

    async def test_ask_response_schema_object(self) -> None:
        agent = Agent("test", config=TestConfig('{"data": 42}'))

        reply = await agent.ask("Hi!", response_schema=ResponseSchema(int, name="MyInt"))
        result = await reply.content()

        assert result == 42

    async def test_ask_callable_override(self) -> None:
        @response_schema
        def double(content: str) -> int:
            return int(content) * 2

        agent = Agent("test", config=TestConfig("21"))

        reply = await agent.ask("Hi!", response_schema=double)
        result = await reply.content()

        assert result == 42

    async def test_ask_prompted_schema_override(self) -> None:
        agent = Agent("test", config=TestConfig('{"data": 42}'))

        reply = await agent.ask("Hi!", response_schema=PromptedSchema(int))
        result = await reply.content()

        assert result == 42

    async def test_ask_overrides_agent_schema(self) -> None:
        agent = Agent("test", config=TestConfig('{"data": 3.14}'), response_schema=int)

        reply = await agent.ask("Hi!", response_schema=float)
        result = await reply.content()

        assert result == 3.14

    async def test_ask_none_drops_schema(self) -> None:
        agent = Agent("test", config=TestConfig("hello"), response_schema=int)

        reply = await agent.ask("Hi!", response_schema=None)
        result = await reply.content()

        assert result == "hello"

    async def test_next_turn_preserves_agent_schema(self) -> None:
        agent = Agent("test", config=TestConfig('{"data": 42}', '{"data": 7}'), response_schema=int)

        reply = await agent.ask("Hi!")
        assert await reply.content() == 42

        next_reply = await reply.ask("Again!")
        assert await next_reply.content() == 7

    async def test_ask_override_does_not_persist(self) -> None:
        agent = Agent("test", config=TestConfig('{"data": 3.14}', "42"))

        reply = await agent.ask("Hi!", response_schema=float)
        assert await reply.content() == 3.14

        next_reply = await reply.ask("Again!")
        assert await next_reply.content() == "42"
