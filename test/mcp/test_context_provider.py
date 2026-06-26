# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import BaseModel

from ag2 import Agent
from ag2.events import ModelResponse
from ag2.mcp.executor import AgentExecutor, AskContext
from ag2.testing import TestConfig


class _Weather(BaseModel):
    city: str


def _text(result: object) -> str:
    block = result[0]  # type: ignore[index]
    return getattr(block, "text", "")


@pytest.mark.asyncio
class TestContextProvider:
    async def test_provider_invoked_and_forwarded(self) -> None:
        seen: dict[str, object] = {}

        async def provider(access: object) -> AskContext:
            seen["called"] = True
            seen["access"] = access
            return AskContext(variables={"x": 1}, prompt="custom system prompt")

        agent = Agent("greeter", config=TestConfig("hi"))
        executor = AgentExecutor(agent, stream_progress=False, context_provider=provider)

        result = await executor.call("ask", message="hello", context=None, request_context=None)

        assert seen["called"] is True
        # No auth context bound in this unit test, so the provider gets None.
        assert seen["access"] is None
        # The reply came back (the injected variables/prompt were accepted by ask()).
        assert _text(result) == "hi"

    async def test_no_provider_is_stateless(self) -> None:
        agent = Agent("greeter", config=TestConfig("hi"))
        executor = AgentExecutor(agent, stream_progress=False)

        result = await executor.call("ask", message="hello", context=None, request_context=None)

        assert _text(result) == "hi"

    async def test_empty_message_is_error(self) -> None:
        executor = AgentExecutor(Agent("greeter", config=TestConfig("hi")), stream_progress=False)

        result = await executor.call("ask", message="", context=None, request_context=None)

        assert result.isError is True  # type: ignore[union-attr]

    async def test_provider_tools_field_forwarded(self) -> None:
        # AskContext.tools set (variables/prompt left None) exercises the tools
        # branch and the skipped variables/prompt branches.
        async def provider(access: object) -> AskContext:
            return AskContext(tools=[])

        executor = AgentExecutor(
            Agent("greeter", config=TestConfig("hi")), stream_progress=False, context_provider=provider
        )

        result = await executor.call("ask", message="hello", context=None, request_context=None)

        assert _text(result) == "hi"

    async def test_structured_output_none_is_error(self) -> None:
        # Object response_schema + an empty model reply -> content() is None ->
        # to_structured_dict is None -> the executor returns an isError result.
        agent = Agent("weather", config=TestConfig(ModelResponse(message=None)), response_schema=_Weather)
        executor = AgentExecutor(agent, stream_progress=False)

        result = await executor.call("ask", message="weather?", context=None, request_context=None)

        assert result.isError is True  # type: ignore[union-attr]
