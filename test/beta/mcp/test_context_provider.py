# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta import Agent
from autogen.beta.mcp.executor import AgentExecutor, AskContext
from autogen.beta.testing import TestConfig


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
