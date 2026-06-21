# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta import Agent
from autogen.beta.events import ModelMessage, ModelResponse, Usage
from autogen.beta.testing import TestConfig


def _response(text: str, prompt: int, completion: int) -> ModelResponse:
    return ModelResponse(
        message=ModelMessage(text),
        usage=Usage(prompt_tokens=prompt, completion_tokens=completion),
        model="claude",
        provider="anthropic",
    )


@pytest.mark.asyncio()
class TestAgentReplyUsage:
    async def test_single_turn(self) -> None:
        agent = Agent("", config=TestConfig(_response("hi", 12, 3)))

        reply = await agent.ask("Hello")

        usage = await reply.usage()
        assert usage.total == Usage(prompt_tokens=12, completion_tokens=3)
        assert usage.by_model == {"claude": Usage(prompt_tokens=12, completion_tokens=3)}

    async def test_accumulates_over_run(self) -> None:
        agent = Agent("", config=TestConfig(_response("first", 10, 2), _response("second", 5, 1)))

        reply = await agent.ask("Hi")
        reply = await reply.ask("Follow up")

        # whole run accumulates all turns on the stream
        usage = await reply.usage()
        assert usage.total == Usage(prompt_tokens=15, completion_tokens=3)
        assert len(usage.records) == 2
